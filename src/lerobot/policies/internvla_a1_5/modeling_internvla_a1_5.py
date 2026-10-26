#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import logging
import math
from collections import deque
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
import torch._dynamo as dynamo
from torch import Tensor, nn

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextModel,
    Qwen3_5Config,
    Qwen3_5Tokenizer,
)

from lerobot.policies.internvla_a1_5.action_tokens import ensure_qwen35_action_tokens
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.wan_model import WanVideoModel
from lerobot.policies.internvla_a1_5.wan.modules.model import sinusoidal_embedding_1d
from lerobot.policies.internvla_a1_5.wan.utils.fm import FlowMatchScheduler
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    LABEL_MODE_FAST,
    LABEL_MODE_NONE,
    LABEL_MODE_TEXT,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import format_big_number
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OBS_PREFIX,
    OBS_STR,
    OPENPI_ATTENTION_MASK_VALUE,
)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    qwen3_5,
    action_expert,
    prefix_len: int,
    knowledge_insulation: bool = False,
    use_sdpa: bool = False,
    linear_attn_mask: torch.Tensor | None = None,
):
    """Run one transformer layer jointly on [VLM prefix, action expert suffix].

    When ``knowledge_insulation`` is True, the suffix (action expert) queries
    attend to the prefix keys/values with gradient detached, so gradients from
    the action / suffix branch cannot flow back into the VLM.

    ``linear_attention`` layers are processed fully independently per model –
    their recurrent state cannot be shared – so KI is naturally enforced there.

    Args:
        linear_attn_mask: 2D [B, total_len] padding mask for linear attention
            layers (1=real, 0=padding/masked). If None, no masking is applied.
    """

    models = [qwen3_5.language_model, action_expert]
    layer_type = qwen3_5.language_model.layers[layer_idx].layer_type

    if layer_type == "linear_attention":
        if linear_attn_mask is not None:
            prefix_linear_mask = linear_attn_mask[:, :prefix_len]
            suffix_linear_mask = linear_attn_mask[:, prefix_len:]
            linear_masks_per_model = [prefix_linear_mask, suffix_linear_mask]
        else:
            linear_masks_per_model = [None, None]

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.linear_attn(
                hidden_states=hidden_states,
                cache_params=None,
                cache_position=None,
                # Linear attention expects a 2D padding mask here.
                attention_mask=linear_masks_per_model[i],
            )
            hidden_states = residual + hidden_states

            after_first_residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype=torch.bfloat16)
            hidden_states = layer.mlp(hidden_states)

            hidden_states = hidden_states + after_first_residual
            outputs_embeds.append(hidden_states)

        return outputs_embeds

    elif layer_type == "full_attention":
        # Compute Q/K/V/gate for each branch separately.
        query_states = []
        key_states = []
        value_states = []
        gates = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]

            q_gate = layer.self_attn.q_proj(hidden_states).view(
                *input_shape, -1, layer.self_attn.head_dim * 2
            )
            query_state, gate = torch.chunk(q_gate, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)

            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_state = layer.self_attn.q_norm(query_state.view(hidden_shape)).transpose(1, 2)
            key_state = layer.self_attn.k_norm(
                layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)
            gates.append(gate)

        prefix_query, suffix_query = query_states
        prefix_key, suffix_key = key_states
        prefix_value, suffix_value = value_states
        prefix_gate, suffix_gate = gates

        # Joint K/V/Q for applying rotary position embedding on the full sequence.
        joint_query = torch.cat(query_states, dim=2)
        joint_key = torch.cat(key_states, dim=2)
        joint_value = torch.cat(value_states, dim=2)

        dummy_tensor = torch.zeros(
            joint_query.shape[0],
            joint_query.shape[2],
            joint_query.shape[-1],
            device=joint_query.device,
            dtype=joint_query.dtype,
        )
        cos, sin = qwen3_5.language_model.rotary_emb(dummy_tensor, position_ids)
        joint_query, joint_key = modeling_qwen3_5.apply_rotary_pos_emb(
            joint_query, joint_key, cos, sin, unsqueeze_dim=1
        )

        # Split back after RoPE.
        prefix_query = joint_query[:, :, :prefix_len]
        suffix_query = joint_query[:, :, prefix_len:]
        prefix_key = joint_key[:, :, :prefix_len]
        suffix_key = joint_key[:, :, prefix_len:]
        prefix_value = joint_value[:, :, :prefix_len]
        suffix_value = joint_value[:, :, prefix_len:]

        scaling = qwen3_5.language_model.layers[layer_idx].self_attn.scaling
        attn_layer = qwen3_5.language_model.layers[layer_idx].self_attn

        batch_size = joint_query.shape[0]

        # --- prefix queries: attend only to prefix K/V (pad mask already
        # forbids prefix -> suffix). Gradient flows normally for VLM.
        prefix_attn_mask = attention_mask[:, :, :prefix_len, :prefix_len]
        if use_sdpa:
            prefix_key_expanded = modeling_qwen3_5.repeat_kv(prefix_key, attn_layer.num_key_value_groups)
            prefix_value_expanded = modeling_qwen3_5.repeat_kv(prefix_value, attn_layer.num_key_value_groups)
            prefix_att_output = F.scaled_dot_product_attention(
                prefix_query, prefix_key_expanded, prefix_value_expanded,
                attn_mask=prefix_attn_mask.to(prefix_query.dtype), scale=scaling,
            )
            prefix_att_output = prefix_att_output.transpose(1, 2).contiguous()
        else:
            prefix_att_output, _ = modeling_qwen3_5.eager_attention_forward(
                attn_layer,
                prefix_query,
                prefix_key,
                prefix_value,
                prefix_attn_mask,
                scaling,
            )

        # --- suffix queries: attend to [prefix (maybe-detached) K/V, suffix K/V].
        if knowledge_insulation:
            prefix_key_for_suffix = prefix_key.detach()
            prefix_value_for_suffix = prefix_value.detach()
        else:
            prefix_key_for_suffix = prefix_key
            prefix_value_for_suffix = prefix_value

        k_for_suffix = torch.cat([prefix_key_for_suffix, suffix_key], dim=2)
        v_for_suffix = torch.cat([prefix_value_for_suffix, suffix_value], dim=2)
        suffix_attn_mask = attention_mask[:, :, prefix_len:, :]

        if use_sdpa:
            k_for_suffix_expanded = modeling_qwen3_5.repeat_kv(k_for_suffix, attn_layer.num_key_value_groups)
            v_for_suffix_expanded = modeling_qwen3_5.repeat_kv(v_for_suffix, attn_layer.num_key_value_groups)
            suffix_att_output = F.scaled_dot_product_attention(
                suffix_query, k_for_suffix_expanded, v_for_suffix_expanded,
                attn_mask=suffix_attn_mask.to(suffix_query.dtype), scale=scaling,
            )
            suffix_att_output = suffix_att_output.transpose(1, 2).contiguous()
        else:
            suffix_att_output, _ = modeling_qwen3_5.eager_attention_forward(
                attn_layer,
                suffix_query,
                k_for_suffix,
                v_for_suffix,
                suffix_attn_mask,
                scaling,
            )

        att_output = torch.cat([prefix_att_output, suffix_att_output], dim=1)

        head_dim = qwen3_5.language_model.layers[layer_idx].self_attn.head_dim
        num_attention_heads = qwen3_5.language_model.layers[layer_idx].self_attn.config.num_attention_heads
        att_output = att_output.reshape(batch_size, -1, num_attention_heads * head_dim)

        gates_joint = torch.cat(gates, dim=1)

        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]

            att_out_slice = att_output[:, start_pos:end_pos]
            gate_slice = gates_joint[:, start_pos:end_pos]

            att_out_slice = att_out_slice * torch.sigmoid(gate_slice)

            if att_out_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                att_out_slice = att_out_slice.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_out_slice)

            out_emb = out_emb + hidden_states
            after_first_residual = out_emb.clone()
            out_emb = layer.post_attention_layernorm(out_emb)

            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)

            out_emb = out_emb + after_first_residual
            outputs_embeds.append(out_emb)
            start_pos = end_pos
        return outputs_embeds

    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")


class ActionExpertConfig:
    """Configuration for the action expert module.

    Note: head_dim is inherited from VLM for proper cross-attention computation.
    Only hidden_size and intermediate_size can be customized.
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        head_dim: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
    ):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


class InternVLAA15WithExpertModel(nn.Module):
    """Qwen3_5 model with action expert for InternVLAA15."""

    def __init__(
        self,
        vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B",
        action_expert_config: ActionExpertConfig | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.qwen3_5 = Qwen3_5ForConditionalGeneration.from_pretrained(vlm_model_name_or_path)
            tokenizer = Qwen3_5Tokenizer.from_pretrained(vlm_model_name_or_path)
            ensure_qwen35_action_tokens(tokenizer, self.qwen3_5)

        vlm_text_config = self.qwen3_5.config.text_config

        if action_expert_config is None:
            action_expert_config = ActionExpertConfig()

        if action_expert_config.head_dim is None:
            action_expert_config.head_dim = vlm_text_config.head_dim
        if action_expert_config.hidden_size is None:
            action_expert_config.hidden_size = vlm_text_config.hidden_size
        if action_expert_config.intermediate_size is None:
            action_expert_config.intermediate_size = vlm_text_config.intermediate_size

        action_expert_config.num_attention_heads = vlm_text_config.num_attention_heads
        action_expert_config.num_key_value_heads = vlm_text_config.num_key_value_heads

        action_expert_config_hf = CONFIG_MAPPING["qwen3_5_text"]()
        action_expert_config_hf.head_dim = action_expert_config.head_dim
        action_expert_config_hf.hidden_size = action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size = action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads = action_expert_config.num_attention_heads
        action_expert_config_hf.num_key_value_heads = action_expert_config.num_key_value_heads
        action_expert_config_hf.num_hidden_layers = vlm_text_config.num_hidden_layers
        action_expert_config_hf.max_position_embeddings = vlm_text_config.max_position_embeddings
        action_expert_config_hf.rope_parameters = vlm_text_config.rope_parameters
        action_expert_config_hf.rms_norm_eps = vlm_text_config.rms_norm_eps

        action_expert_config_hf.layer_types = vlm_text_config.layer_types
        action_expert_config_hf.linear_conv_kernel_dim = vlm_text_config.linear_conv_kernel_dim
        action_expert_config_hf.linear_key_head_dim = vlm_text_config.linear_key_head_dim
        action_expert_config_hf.linear_value_head_dim = vlm_text_config.linear_value_head_dim
        action_expert_config_hf.linear_num_key_heads = vlm_text_config.linear_num_key_heads
        action_expert_config_hf.linear_num_value_heads = vlm_text_config.linear_num_value_heads

        self.action_expert = Qwen3_5TextModel(config=action_expert_config_hf)
        self.action_expert.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        knowledge_insulation: bool = False,
        use_sdpa: bool = False,
        linear_attn_mask: torch.Tensor | None = None,
    ):
        if inputs_embeds[1] is None:
            prefix_output = self.qwen3_5.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.action_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.qwen3_5.language_model, self.action_expert]
            num_layers = self.qwen3_5.config.text_config.num_hidden_layers

            prefix_len = inputs_embeds[0].shape[1]

            use_gradient_checkpointing = (
                hasattr(self.action_expert, "gradient_checkpointing")
                and self.action_expert.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_5,
                        self.action_expert,
                        prefix_len,
                        knowledge_insulation,
                        use_sdpa,
                        linear_attn_mask,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        qwen3_5=self.qwen3_5,
                        action_expert=self.action_expert,
                        prefix_len=prefix_len,
                        knowledge_insulation=knowledge_insulation,
                        use_sdpa=use_sdpa,
                        linear_attn_mask=linear_attn_mask,
                    )

            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class LoRALinear(nn.Module):
    """nn.Linear 드롭인 대체(연구 기능): base(동결) + 저랭크 A,B(학습).

    out = base(x) + scaling * B(A(dropout(x)))
    - B를 zero-init → 생성 시점엔 base와 완전히 동일(항등). 학습하며 VLM 적응.
    - `.weight`/`.bias` 프록시로 커스텀 forward의 `.weight.dtype` 참조와 호환.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, r, bias=False).to(dev, dt)
        self.lora_B = nn.Linear(r, base.out_features, bias=False).to(dev, dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # 항등 시작
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # 커스텀 forward가 q_proj.weight.dtype 등을 참조하므로 프록시 제공
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # 원본 체크포인트({prefix}weight/{prefix}bias)를 base로 매핑 → 사전학습 가중치 로드.
        # 이미 wrap된(base.weight 키) 체크포인트에는 no-op.
        for suffix in ("weight", "bias"):
            k = prefix + suffix
            if k in state_dict:
                state_dict[prefix + "base." + suffix] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))


def inject_vlm_lora(vlm: nn.Module, targets, r: int, alpha: int, dropout: float) -> int:
    """vlm 내부의 대상 nn.Linear를 in-place로 LoRALinear로 교체. 교체 수 반환.

    모듈 구조·경로는 그대로 유지되므로 커스텀 forward(layer.self_attn.q_proj 직접 호출)와 호환.
    """
    targets = set(targets)
    replaced = 0
    for module in vlm.modules():
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced += 1
    return replaced


class SeamGroundingModule(nn.Module):
    """TurboVLA 이식(연구 기능): prefix 임베딩의 시각↔지시 토큰을 얕은 양방향
    cross-attn으로 명시 정렬해 grounded prefix를 만든다. 각 토큰은 '다른 모달리티'
    토큰에만 attend(교차모달 전용). config.use_seam_grounding=True 일 때만 생성/사용.

    출력 = 원본 embs + delta 형태. o/ffn2를 0으로 초기화하므로 학습 초반 delta≈0 →
    원본 동작 그대로에서 점진적으로 grounding을 반영. detach_input=True면 delta가
    detach된 입력에서 계산되어 seam gradient가 VLM으로 역류하지 않음(KI 유지).
    """

    def __init__(self, hidden_size, num_layers=4, num_heads=8, detach_input=True):
        super().__init__()
        self.num_heads = num_heads
        self.detach_input = detach_input
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_size),
                        "q": nn.Linear(hidden_size, hidden_size),
                        "k": nn.Linear(hidden_size, hidden_size),
                        "v": nn.Linear(hidden_size, hidden_size),
                        "o": nn.Linear(hidden_size, hidden_size),
                        "ffn_norm": nn.LayerNorm(hidden_size),
                        "ffn1": nn.Linear(hidden_size, 4 * hidden_size),
                        "ffn2": nn.Linear(4 * hidden_size, hidden_size),
                    }
                )
            )
        for lyr in self.layers:  # zero-init 출력 → 초기 delta≈0 (원본 보존)
            nn.init.zeros_(lyr["o"].weight)
            nn.init.zeros_(lyr["o"].bias)
            nn.init.zeros_(lyr["ffn2"].weight)
            nn.init.zeros_(lyr["ffn2"].bias)

    def forward(self, embs, visual_mask, pad_mask):
        # embs: [B, L, D] / visual_mask, pad_mask: [B, L] bool
        src = embs.detach() if self.detach_input else embs
        B, L, D = src.shape
        H, hd = self.num_heads, D // self.num_heads
        vm = visual_mask & pad_mask          # 유효 시각 토큰
        tm = (~visual_mask) & pad_mask       # 유효 지시(텍스트) 토큰
        # 교차모달 허용: 서로 다른 모달리티 & 둘 다 유효
        allowed = (vm.unsqueeze(2) & tm.unsqueeze(1)) | (tm.unsqueeze(2) & vm.unsqueeze(1))
        no_partner = ~allowed.any(dim=2)     # 교차모달 상대 없음(패딩/단일모달)
        eye = torch.eye(L, dtype=torch.bool, device=src.device).unsqueeze(0)
        allowed = allowed | (no_partner.unsqueeze(2) & eye)  # NaN 방지(뒤에서 원본으로 덮음)
        attn_bias = torch.zeros(B, 1, L, L, dtype=src.dtype, device=src.device)
        attn_bias = attn_bias.masked_fill(~allowed.unsqueeze(1), float("-inf"))

        x = src
        for lyr in self.layers:
            h = lyr["norm"](x)
            q = lyr["q"](h).view(B, L, H, hd).transpose(1, 2)
            k = lyr["k"](h).view(B, L, H, hd).transpose(1, 2)
            v = lyr["v"](h).view(B, L, H, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            a = a.transpose(1, 2).reshape(B, L, D)
            x = x + lyr["o"](a)
            h2 = lyr["ffn_norm"](x)
            x = x + lyr["ffn2"](F.gelu(lyr["ffn1"](h2)))

        delta = x - src                      # grounding delta (detach_input이면 VLM 무관)
        out = embs + delta                   # 원본 identity 경로 유지 → VLM 학습 보존
        out = torch.where(no_partner.unsqueeze(-1), embs, out)  # 상대 없던 토큰은 원본
        return out


class ForesightResidualPrior(nn.Module):
    """Foresight Residual Prior (DRIFT 변형 · 연구 기능 · config.use_residual_prior 일 때만 생성).

    foresight 토큰 출력(예측된 미래 잠재)을 mean-pool 한 벡터와 flow 시각 t 를 작은 MLP 에 넣어
    [B, chunk, action_dim] 잔차 r 를 만들고, expert 의 velocity 예측에 더한다:  v_hat = v_pred + r.
    DRIFT 와 달리 prior 의 입력이 VLM 임베딩이 아니라 '예측된 미래' 이므로 역동역학 prior 에 해당한다.

    [09-14 정정] 초기 주석에 "x-공간에서 −t·r 로 들어가 t→0 에서 사라지는 time-gated prior" 라고
    적었으나 이는 틀렸다. 추론의 Euler 루프가 매 스텝 x ← x + dt·(v_pred + r) 로 r 을 누적하므로
    (dt=−0.1, 10 스텝) 최종적으로 −Σ0.1·r ≈ −r 이 x 에 그대로 남는다. t→0 에서 사라지지 않는다.
    출력층 zero-init → 생성 시점에 r ≡ 0 (원본과 항등). gradient 는 foresight 토큰으로 흘러
    co-training 된다(의도적으로 detach 하지 않음).
    """

    def __init__(self, hidden: int, chunk_size: int, action_dim: int,
                 min_period: float, max_period: float, mlp_hidden: int = 0, state_dim: int = 0):
        super().__init__()
        mlp_hidden = mlp_hidden or hidden
        self.hidden, self.chunk_size, self.action_dim = hidden, chunk_size, action_dim
        self.min_period, self.max_period = min_period, max_period
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, chunk_size * action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)  # 항등 시작
        nn.init.zeros_(self.net[-1].bias)
        # ablation(source=state): raw proprioception [B, state_dim] → hidden. foresight 경로에선 미사용.
        self.state_proj = nn.Linear(state_dim, hidden) if state_dim else None

    def forward(self, pooled: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # pooled: [B, hidden]  timestep: [B]
        t_emb = create_sinusoidal_pos_embedding(
            timestep.float(), self.hidden, self.min_period, self.max_period, device=timestep.device
        )
        w_dtype = self.net[0].weight.dtype
        r = self.net(torch.cat([pooled.to(w_dtype), t_emb.to(w_dtype)], dim=-1))
        return r.view(pooled.shape[0], self.chunk_size, self.action_dim)


class LatentForesightDecoder(nn.Module):
    """v2 Latent Foresight 디코더: N개 학습 query가 foresight 토큰에 cross-attn(pre-norm)
    → 얼린 인코더(DINOv2)의 patch feature grid [B, N, out_dim]을 예측. mean-pool로 뭉개는
    v1 MLP와 달리 공간구조를 보존한다."""

    def __init__(self, hidden: int, n_query: int, out_dim: int, n_layers: int = 3, n_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_query, hidden))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(hidden),
                "xattn": nn.MultiheadAttention(hidden, n_heads, batch_first=True),
                "norm_f": nn.LayerNorm(hidden),
                "ffn": nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)),
            })
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, foresight: torch.Tensor) -> torch.Tensor:  # [B, L, hidden]
        B = foresight.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        for lyr in self.layers:
            a, _ = lyr["xattn"](lyr["norm_q"](q), foresight, foresight, need_weights=False)
            q = q + a
            q = q + lyr["ffn"](lyr["norm_f"](q))
        return self.out(q)  # [B, n_query, out_dim]


class InternVLAA15(nn.Module):

    def __init__(self, config: InternVLAA15Config):
        super().__init__()
        self.config = config

        action_expert_config = ActionExpertConfig(
            hidden_size=config.action_expert_hidden_size,
            intermediate_size=config.action_expert_intermediate_size,
        )

        self.qwen3_5_with_expert = InternVLAA15WithExpertModel(
            vlm_model_name_or_path=config.vlm_model_name_or_path,
            action_expert_config=action_expert_config,
            precision=config.dtype,
        )

        action_expert_hidden_size = self.qwen3_5_with_expert.action_expert.config.hidden_size

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_hidden_size)
        self.action_out_proj = nn.Linear(action_expert_hidden_size, config.max_action_dim)

        if not self.config.tokenize_state:
            self.state_proj = nn.Linear(config.max_state_dim, action_expert_hidden_size)

        self.action_time_mlp_in = nn.Linear(2 * action_expert_hidden_size, action_expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_hidden_size, action_expert_hidden_size)


        self.learnable_tokens = nn.Parameter(
            torch.zeros(config.num_learnable_tokens, action_expert_hidden_size)
        )
        nn.init.trunc_normal_(self.learnable_tokens, std=0.02)
        self.learnable_tokens_in_proj = nn.Linear(
            action_expert_hidden_size, action_expert_hidden_size
        )

        if not config.action_loss_only and not getattr(config, "use_latent_foresight", False):
            self.wan_video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision,
            )
            # WAN을 별도 GPU에 둘 목표 device(첫 forward에서 배치, 학습 본체 .to() 이후).
            self._wan_target_device = getattr(config, "wan_device", "") or None
            wan_dim = self.wan_video_model.wan_model.dim
            self.learnable_to_wan_proj = nn.Linear(action_expert_hidden_size, wan_dim)
            self.fm_video_scheduler = FlowMatchScheduler(
                shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000,
            )
            self.fm_video_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            lat_T = 1 + config.num_video_frames // 4
            lat_H = config.video_height // 32
            lat_W = config.video_width // 32
            self.register_buffer(
                "_wan_grid_sizes", torch.tensor([lat_T, lat_H, lat_W], dtype=torch.long)
            )

        # ---- Seam Grounding (연구 기능, flag일 때만 생성 → off면 파라미터·VRAM 0) ----
        if getattr(config, "use_seam_grounding", False):
            vlm_hidden = self.qwen3_5_with_expert.qwen3_5.get_input_embeddings().weight.shape[-1]
            self.seam_grounding = SeamGroundingModule(
                hidden_size=vlm_hidden,
                num_layers=config.seam_num_layers,
                num_heads=config.seam_num_heads,
                detach_input=config.seam_detach_input,
            )
            # 모델 dtype에 맞춰 캐스팅 (나머지 모델과 동일 정밀도)
            _seam_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
            self.seam_grounding = self.seam_grounding.to(_seam_dtype)

        # ---- Latent Foresight (연구 기능, flag일 때만 생성 → off면 파라미터·VRAM 0) ----
        # foresight 토큰(N개)을 mean-pool → 미래 프레임의 coarse 표현(3·g·g)으로 회귀.
        # 얼린 WAN 대신 쓰는 경량 head. WAN 미로드로 학습 VRAM −10GB.
        if getattr(config, "use_latent_foresight", False):
            lf_target = getattr(config, "latent_foresight_target", "pixel")
            if lf_target == "dino":
                # v2: 얼린 DINOv2를 target 인코더로 → 미래 프레임의 semantic patch feature 회귀.
                from transformers import Dinov2Model
                self.lf_dino = Dinov2Model.from_pretrained(config.latent_foresight_dino_name)
                self.lf_dino.eval()
                for p in self.lf_dino.parameters():
                    p.requires_grad = False
                dino_dim = self.lf_dino.config.hidden_size
                n_patch = (224 // self.lf_dino.config.patch_size) ** 2
                # video[-1,1] → [0,1] → ImageNet 정규화 버퍼
                self.register_buffer("_lf_dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("_lf_dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                self.latent_foresight_decoder = LatentForesightDecoder(
                    action_expert_hidden_size, n_query=n_patch, out_dim=dino_dim,
                    n_layers=config.latent_foresight_decoder_layers,
                )
            else:
                # v1: coarse 픽셀 MLP head
                g = config.latent_foresight_grid
                self.latent_foresight_head = nn.Sequential(
                    nn.Linear(action_expert_hidden_size, action_expert_hidden_size),
                    nn.GELU(),
                    nn.Linear(action_expert_hidden_size, 3 * g * g),
                )

        # ---- Spatial Query Tokens (9.29 · 기하 쿼리 토큰, flag일 때만 생성 → off면 파라미터·VRAM 0) ----
        # LF v2 와 같은 구조(학습 슬롯 → cross-attn 디코더 → 얼린 교사 특징 회귀)를 하나 더 두되,
        # 교사를 '미래 프레임의 DINOv2(의미)' 가 아니라 '현재 프레임의 VGGT(3D 기하)' 로 둔다.
        # suffix = [state?][spatial×N][foresight×50][action×chunk]. 교사·디코더는 학습 전용, 추론 때는 토큰만 남는다.
        if getattr(config, "use_spatial_tokens", False):
            n_sp = config.spatial_num_tokens
            self.spatial_tokens = nn.Parameter(torch.zeros(n_sp, action_expert_hidden_size))
            nn.init.trunc_normal_(self.spatial_tokens, std=0.02)
            # 격자 대응 학습 위치 임베딩 (query 가 화면의 어느 칸인지 고정) — zero-init
            self.spatial_pos_embed = nn.Parameter(torch.zeros(n_sp, action_expert_hidden_size))
            self.spatial_tokens_in_proj = nn.Linear(action_expert_hidden_size, action_expert_hidden_size)
            _teacher = getattr(config, "spatial_teacher", "vggt")
            if _teacher == "vggt":
                # aggregator 토큰 2048 (frame-attn ‖ global-attn) / 포인트 헤드 점 지도 xyz 3
                sp_dim = 3 if getattr(config, "spatial_vggt_target", "tokens") == "points" else 2048
            elif _teacher == "dino":
                if not hasattr(self, "lf_dino"):
                    raise ValueError("spatial_teacher='dino' 는 use_latent_foresight + latent_foresight_target='dino' 에서만 (lf_dino 재사용)")
                sp_dim = self.lf_dino.config.hidden_size
            else:
                raise ValueError(f"spatial_teacher={_teacher}")
            self.spatial_decoder = LatentForesightDecoder(
                action_expert_hidden_size, n_query=n_sp, out_dim=sp_dim,
                n_layers=getattr(config, "spatial_decoder_layers", 3),
            )
            # 교사는 리스트에 담아 nn.Module 등록을 피한다 → state_dict·체크포인트 제외, 첫 학습 forward 에서 지연 로드(추론 때 미로드)
            self._spatial_teacher = []
            self._spatial_calls = 0

        # ---- Foresight Residual Prior (연구 기능, flag일 때만 생성 → off면 파라미터·VRAM 0) ----
        if getattr(config, "use_residual_prior", False):
            self.residual_prior = ForesightResidualPrior(
                hidden=action_expert_hidden_size,
                chunk_size=config.chunk_size,
                action_dim=config.max_action_dim,
                min_period=config.min_period,
                max_period=config.max_period,
                mlp_hidden=getattr(config, "residual_prior_hidden", 0),
                state_dim=config.max_state_dim if getattr(config, "residual_prior_source", "foresight") == "state" else 0,
            )
            logging.info(
                f"[ResidualPrior] enabled: source={config.residual_prior_source} "
                f"params={sum(p.numel() for p in self.residual_prior.parameters())/1e6:.2f}M"
            )

        # ---- VLM LoRA (연구 기능, flag일 때만 주입 → off면 파라미터·VRAM 0) ----
        if getattr(config, "use_vlm_lora", False):
            n = inject_vlm_lora(
                self.qwen3_5_with_expert.qwen3_5,
                targets=config.lora_targets,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
            logging.info(f"[VLM-LoRA] injected LoRA into {n} Linear modules (r={config.lora_r})")

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, dynamic=True, mode=config.compile_mode)

        self.set_requires_grad()
        self._setup_wan_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwen3_5_with_expert.qwen3_5.visual.eval()
            for params in self.qwen3_5_with_expert.qwen3_5.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwen3_5_with_expert.qwen3_5.eval()
            for params in self.qwen3_5_with_expert.qwen3_5.parameters():
                params.requires_grad = False

        # VRAM 절감(연구 학습): VLM 백본 동결 → seam/expert/head만 학습. 기본 off.
        if getattr(self.config, "freeze_vlm", False):
            self.qwen3_5_with_expert.qwen3_5.eval()
            for params in self.qwen3_5_with_expert.qwen3_5.parameters():
                params.requires_grad = False

        # VLM LoRA(연구 학습): base 동결 + 저랭크 adapter(A,B)만 학습. freeze_vlm보다 뒤에 두어 override.
        if getattr(self.config, "use_vlm_lora", False):
            for name, params in self.qwen3_5_with_expert.qwen3_5.named_parameters():
                params.requires_grad = ("lora_A" in name) or ("lora_B" in name)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.qwen3_5_with_expert.qwen3_5.language_model.gradient_checkpointing = True
        self.qwen3_5_with_expert.qwen3_5.visual.gradient_checkpointing = True
        self.qwen3_5_with_expert.action_expert.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for InternVLAA15 model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_5_with_expert.qwen3_5.language_model.gradient_checkpointing = False
        self.qwen3_5_with_expert.qwen3_5.visual.gradient_checkpointing = False
        self.qwen3_5_with_expert.action_expert.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for InternVLAA15 model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _compute_fast_token_mask(
        self, lang_tokens: Tensor, provided_mask: Tensor | None = None
    ) -> Tensor:
        """Return a bool mask marking fast-token positions in the prefix.

        Uses the transform-provided mask if available; otherwise derives it from
        the configured ``[action_token_min, action_token_max]`` range.
        """
        if provided_mask is not None:
            return provided_mask.to(device=lang_tokens.device, dtype=torch.bool)
        return (
            (lang_tokens >= self.config.action_token_min)
            & (lang_tokens <= self.config.action_token_max)
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        alpha = self.config.time_sampling_beta_alpha
        beta = self.config.time_sampling_beta_beta
        stratified = getattr(self.config, "time_sampling_stratified", False)
        if stratified and abs(beta - 1.0) < 1e-8:
            # 제안 2a: Beta(α,1) 의 CDF 는 F(x)=x^α → 역CDF 는 u^{1/α}.
            # 배치 내 t 를 iid 로 뽑는 대신 [0,1) 을 bsize 등분해 한 칸에 하나씩(층화추출)
            # 뽑아 시간 커버리지의 배치 간 편차를 없앤다. 기댓값 분포는 Beta(α,1) 그대로.
            strata = torch.arange(bsize, device=device, dtype=torch.float32)
            u = (strata + torch.rand(bsize, device=device)) / bsize
            u = u[torch.randperm(bsize, device=device)]  # 배치 순서와 t 크기의 상관 제거
            time_beta = u.pow(1.0 / alpha)
        else:
            if stratified:
                logging.warning(
                    "[time-sampling] stratified 는 beta_beta==1 에서만 닫힌 역CDF 가 존재 "
                    f"(현재 {beta}) → iid 샘플링으로 폴백"
                )
            time_beta = sample_beta(alpha, beta, bsize, device)
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # action loss 성형 (제안 1 · 제안 3) — 파라미터 0개, 손실 텐서만 변형
    # ------------------------------------------------------------------

    def _criticality_weight(self, actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor | None:
        """제안 1: GT 속도 기반 타임스텝 가중 w [B, chunk]. off 면 None (곱셈 자체를 생략)."""
        mode = getattr(self.config, "action_loss_weighting", "none")
        if mode == "none":
            return None
        dims = getattr(self.config, "alw_speed_dims", 0) or actions.shape[-1]
        dims = min(dims, actions.shape[-1])
        eps = getattr(self.config, "alw_eps", 1e-3)
        # gripper(이진 차원)를 빼고 연속 운동 차원만으로 속도를 잰다 (AttenA+ 와 동일한 이유).
        speed = actions[..., :dims].float().norm(dim=-1).clamp_min(eps)  # [B, chunk]

        fn = getattr(self.config, "alw_fn", "inv")
        if fn == "inv":
            w = 1.0 / speed
        elif fn == "inv_sq":
            w = 1.0 / speed.pow(2)
        elif fn == "exp":
            w = torch.exp(-getattr(self.config, "alw_alpha", 5.0) * speed)
        elif fn == "log":
            w = 1.0 / torch.log1p(speed).clamp_min(eps)
        else:
            raise ValueError(f"alw_fn={fn}")

        clip = max(float(getattr(self.config, "alw_clip", 3.0)), 1.0)
        w = w.clamp(1.0 / clip, clip)

        if mode == "inv_speed_tgate":
            # 우리 변형: 저노이즈(t→0, 세부 정밀화) 구간에서만 가중을 살리고
            # 고노이즈(t→1, 모드 선택) 구간에선 w→1 로 원래 손실에 수렴시킨다.
            w = 1.0 + (w - 1.0) * (1.0 - timestep.float().clamp(0.0, 1.0))[:, None]
        elif mode != "inv_speed":
            raise ValueError(f"action_loss_weighting={mode}")

        if getattr(self.config, "alw_normalize", True):
            w = w * (w.numel() / w.sum().clamp_min(eps))  # 배치 평균 1 → 유효 학습률 보존
        return w

    def _shape_action_loss(self, v_t: torch.Tensor, u_t: torch.Tensor,
                           actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """MSE 에 제안 3(Sobolev) 을 더하고 제안 1(가중) 을 곱한 [B, chunk, dim] 손실."""
        err = v_t - u_t
        loss = err.pow(2)

        lam = float(getattr(self.config, "action_loss_sobolev_weight", 0.0))
        if lam > 0.0 and loss.shape[1] > 1:
            # 제안 3: 속도 오차의 chunk 방향 1차 차분에도 벌점 → H¹(Sobolev) 가중.
            # 마지막 스텝은 차분 대상이 없어 0 을 채워 [B, chunk, dim] 모양을 유지한다.
            d = err[:, 1:] - err[:, :-1]
            d = torch.cat([d, torch.zeros_like(d[:, :1])], dim=1)
            loss = loss + lam * d.pow(2)

        w = self._criticality_weight(actions, timestep)
        if w is not None:
            loss = loss * w[:, :, None]
        return loss

    @dynamo.disable
    def embed_prefix(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks, labels=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_5_with_expert.qwen3_5.config.image_token_id
        D1 = pixel_values.shape[-1]
        pixel_values = pixel_values.view(-1, D1)
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs = self.qwen3_5_with_expert.qwen3_5.visual(pixel_values, image_grid_thw).pooler_output

        embs = self.qwen3_5_with_expert.qwen3_5.get_input_embeddings()(lang_tokens)
        B, L, D2 = embs.shape
        embs = embs.view(-1, D2)
        lang_tokens = lang_tokens.view(-1)
        embs[lang_tokens == image_token_id] = image_embs
        embs = embs.view(B, L, D2)

        pad_masks = lang_masks.to(torch.bool)
        # Force causal attention over the prefix to match Qwen3.5's pretrained
        # decoder-only LM regime. Each valid position starts its own block, so
        # cumsum becomes 1..L and `cumsum_kv <= cumsum_q` reduces to a standard
        # lower-triangular causal mask. The `labels` argument is kept for API
        # compatibility but no longer alters the prefix attention pattern.
        del labels  # noqa: F841 — intentionally unused
        att_masks = pad_masks.clone()

        # ---- Seam Grounding (연구 기능, flag일 때만) : H_t 원료(prefix 임베딩)의
        # 시각↔지시 토큰을 양방향 cross-attn으로 명시 정렬. off면 이 분기 자체를 건너뜀. ----
        if getattr(self.config, "use_seam_grounding", False):
            visual_mask = (lang_tokens == image_token_id).view(B, L)
            embs = self.seam_grounding(embs, visual_mask, pad_masks)

        return embs, pad_masks, att_masks

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks):
        L = lang_tokens.shape[1]
        pseudo_action_token_id = 777
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * pseudo_action_token_id
        padded_lang_tokens[:, :L] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        position_ids, rope_deltas = self.qwen3_5_with_expert.qwen3_5.model.get_rope_index(
            padded_lang_tokens,
            image_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def _block_suffix_attend_prefix_tokens(
        self,
        att_2d_masks: Tensor,
        prefix_len: int,
        blocked_prefix_mask: Tensor | None,
    ) -> Tensor:
        """Block suffix (action) queries from attending selected prefix keys."""
        if blocked_prefix_mask is None:
            return att_2d_masks
        if blocked_prefix_mask.dtype != torch.bool:
            blocked_prefix_mask = blocked_prefix_mask.bool()
        blocked_prefix_mask = blocked_prefix_mask.to(device=att_2d_masks.device)
        att_2d_masks[:, prefix_len:, :prefix_len] &= ~blocked_prefix_mask[:, None, :]
        return att_2d_masks

    @torch.no_grad()
    def generate_subtask_tokens(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        max_new_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        generated = self.qwen3_5_with_expert.qwen3_5.generate(
            input_ids=lang_tokens,
            attention_mask=lang_masks,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        new_tokens = generated[:, lang_tokens.shape[1] :]
        if new_tokens.numel() == 0:
            return lang_tokens, lang_masks

        appended_masks = torch.ones_like(new_tokens, dtype=lang_masks.dtype, device=lang_masks.device)
        lang_tokens = torch.cat([lang_tokens, new_tokens], dim=1)
        lang_masks = torch.cat([lang_masks, appended_masks], dim=1)
        return lang_tokens, lang_masks

    @torch.no_grad()
    def sample_actions(
        self,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        fast_token_mask: Tensor | None = None,
        noise=None,
        num_steps=None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, rope_deltas = self.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.qwen3_5_with_expert.qwen3_5.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.qwen3_5_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        if self.config.block_action_attend_fast_tokens:
            fast_mask = self._compute_fast_token_mask(lang_tokens, fast_token_mask)
        else:
            fast_mask = None

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                max_prefix_position_ids,
                x_t.to(dtype),
                expanded_time.to(dtype),
                fast_mask=fast_mask,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids,
        x_t,
        timestep,
        fast_mask: Tensor | None = None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Block suffix queries from attending to fast tokens at inference time too.
        if fast_mask is not None:
            mask_b = fast_mask.to(device=full_att_2d_masks.device, dtype=torch.bool)
            full_att_2d_masks[:, :, :prefix_len] &= ~mask_b[:, None, :]

        full_att_2d_masks = self._isolate_spatial(full_att_2d_masks, row_off=0, col_off=prefix_len)
        position_ids = (
            self._suffix_position_ids_1d(suffix_len).repeat(3, 1, 1).to(max_prefix_position_ids)
            + max_prefix_position_ids
        )

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_5_with_expert.action_expert.config._attn_implementation = "eager"

        outputs_embeds, _ = self.qwen3_5_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        suffix_out = outputs_embeds[1]
        action_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v = self.action_out_proj(action_out)
        v, _ = self._apply_residual_prior(v, suffix_out, timestep, state=state)
        return v

    def _setup_wan_grad(self):
        if self.config.action_loss_only or self.config.freeze_learnable_tokens:
            self.learnable_tokens.requires_grad = False
            for p in self.learnable_tokens_in_proj.parameters():
                p.requires_grad = False
        if not hasattr(self, "wan_video_model"):
            return  # WAN not loaded (action_loss_only or use_latent_foresight); nothing else to configure
        for p in self.wan_video_model.vae.model.parameters():
            p.requires_grad = False
        if self.config.freeze_wan_dit:
            for p in self.wan_video_model.wan_model.parameters():
                p.requires_grad = False
        if self.config.freeze_learnable_tokens:
            for p in self.learnable_to_wan_proj.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)

        if self.config.freeze_vision_encoder:
            self.qwen3_5_with_expert.qwen3_5.visual.eval()

        if self.config.train_expert_only:
            self.qwen3_5_with_expert.qwen3_5.eval()
        if not hasattr(self, "wan_video_model"):
            return self  # WAN not loaded (action_loss_only or use_latent_foresight)
        self.wan_video_model.vae.model.eval()
        if self.config.freeze_wan_dit:
            self.wan_video_model.wan_model.eval()
        return self

    # ------------------------------------------------------------------
    # Suffix embedding with learnable tokens
    # ------------------------------------------------------------------

    def _action_time_embs(self, noisy_actions, timestep):
        """[B, chunk, hidden] action+time 토큰. embed_suffix 와 다중 표본 경로가 공유."""
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(lambda a: self.action_in_proj(a), noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        return self._apply_checkpoint(
            lambda x: self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x))),
            action_time_emb,
        )

    def embed_suffix_multi(self, state, x_t_list, time_list):
        """제안 2b: suffix 에 action 블록을 K개 나란히 붙인다.

        [state] [foresight × N] [action^0 × chunk] [action^1 × chunk] ...
        블록 0 은 단일 표본 경로와 토큰·position id 가 동일하고, 블록 간 attention 은
        호출부에서 `_isolate_action_blocks` 로 차단하므로 K=1 결과와 블록 0 이 일치한다.
        """
        embs, pad_masks, att_masks = self.embed_suffix(state, x_t_list[0], time_list[0])
        if len(x_t_list) == 1:
            return embs, pad_masks, att_masks

        bsize = embs.shape[0]
        chunk = self.config.chunk_size
        extra_embs, extra_att = [], []
        for x_t, t in zip(x_t_list[1:], time_list[1:]):
            extra_embs.append(self._action_time_embs(x_t, t))
            extra_att += [1] + [0] * (chunk - 1)  # 각 블록이 새 attention block 을 연다

        extra_embs = torch.cat(extra_embs, dim=1).to(embs.dtype)
        n_extra = extra_embs.shape[1]
        embs = torch.cat([embs, extra_embs], dim=1)
        pad_masks = torch.cat(
            [pad_masks, torch.ones(bsize, n_extra, dtype=torch.bool, device=pad_masks.device)], dim=1
        )
        extra_att = torch.tensor(extra_att, dtype=att_masks.dtype, device=att_masks.device)
        att_masks = torch.cat([att_masks, extra_att[None, :].expand(bsize, n_extra)], dim=1)
        return embs, pad_masks, att_masks

    def _isolate_action_blocks(self, att_2d_masks, first_action_idx, chunk, k_blocks):
        """다중 표본 블록끼리 서로 보지 못하게 막는다 (각 블록은 prefix·state·foresight 만 참조).

        이 차단이 없으면 cumsum 규칙상 뒤 블록이 앞 블록의 x_t 를 보게 되어 표본이 독립이 아니게 된다.
        """
        if k_blocks <= 1:
            return att_2d_masks
        total = k_blocks * chunk
        idx = torch.arange(total, device=att_2d_masks.device) // chunk  # 블록 번호
        same_block = idx[:, None] == idx[None, :]
        sl = slice(first_action_idx, first_action_idx + total)
        att_2d_masks = att_2d_masks.clone()
        att_2d_masks[:, sl, sl] &= same_block[None]
        return att_2d_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Build suffix: [state(1)] [learnable(N)] [action_time(chunk_size)]."""
        embs = []
        pad_masks = []
        att_masks = []

        # State token
        if not self.config.tokenize_state:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self._apply_checkpoint(lambda s: self.state_proj(s), state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        bsize = state.shape[0]
        device = state.device

        # Spatial query tokens (9.29) — state 뒤, foresight 앞. 별도 attention 블록 (off 면 0 개)
        n_sp = self._n_spatial()
        if n_sp > 0:
            sp_emb = self._apply_checkpoint(
                lambda t: self.spatial_tokens_in_proj(t), self.spatial_tokens + self.spatial_pos_embed
            )
            embs.append(sp_emb[None].expand(bsize, -1, -1))
            pad_masks.append(torch.ones(bsize, n_sp, dtype=torch.bool, device=device))
            att_masks += [1] + [0] * (n_sp - 1)

        # Learnable tokens
        num_lt = self.config.num_learnable_tokens
        lt_emb = self._apply_checkpoint(
            lambda t: self.learnable_tokens_in_proj(t), self.learnable_tokens
        )
        lt_emb = lt_emb[None].expand(bsize, -1, -1)
        embs.append(lt_emb)
        pad_masks.append(torch.ones(bsize, num_lt, dtype=torch.bool, device=device))
        att_masks += [1] + [0] * (num_lt - 1)

        # Action + time tokens
        action_time_emb = self._action_time_embs(noisy_actions, timestep)

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + [0] * (self.config.chunk_size - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _residual_prior_term(self, suffix_out: torch.Tensor, timestep: torch.Tensor,
                             state: torch.Tensor | None = None) -> torch.Tensor:
        """prior MLP 출력 [B, chunk, dim]. velocity 모드에선 잔차 r, anchored 모드에선 coarse 행동 g."""
        start = (0 if self.config.tokenize_state else 1) + self._n_spatial()
        if getattr(self.config, "residual_prior_source", "foresight") == "state":
            # ablation 대조군: '예측된 미래' 대신 '현재 관측(proprioception)' 을 prior 입력으로 (LAFM/DRIFT 유형)
            if state is None:
                raise ValueError("residual_prior_source='state' 인데 state 가 전달되지 않음")
            pooled = self.residual_prior.state_proj(state.to(self.residual_prior.state_proj.weight.dtype))
        else:
            pooled = suffix_out[:, start:start + self.config.num_learnable_tokens].mean(dim=1)
        return self.residual_prior(pooled.to(torch.float32), timestep)

    def _apply_residual_prior(self, v: torch.Tensor, suffix_out: torch.Tensor, timestep: torch.Tensor,
                              state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """use_residual_prior 가 켜져 있으면 (v_hat, g_or_r) 를, 아니면 (v, None) 반환 (off 면 연산 0).

        velocity 모드: v_hat = v + r.   anchored 모드: v_hat = v − g  (flow 는 a − g 잔차만 담당).
        suffix 배치: [state(tokenize_state=False 일 때만 1)] [foresight × N] [action × chunk].
        """
        if not getattr(self.config, "use_residual_prior", False):
            return v, None
        term = self._residual_prior_term(suffix_out, timestep, state)
        if getattr(self.config, "residual_prior_mode", "velocity") == "anchored":
            return v - term.to(v.dtype), term
        return v + term.to(v.dtype), term

    # ------------------------------------------------------------------
    # Spatial Query Tokens (9.29) — 위치·마스크·손실 헬퍼 (off 면 전부 항등)
    # ------------------------------------------------------------------
    def _n_spatial(self) -> int:
        if not getattr(self.config, "use_spatial_tokens", False):
            return 0
        return int(getattr(self.config, "spatial_num_tokens", 0))

    def _n_state_tokens(self) -> int:
        return 0 if self.config.tokenize_state else 1

    def _suffix_position_ids_1d(self, suffix_len: int) -> torch.Tensor:
        """suffix 토큰의 1-based rope 위치열. spatial 토큰은 [state?] 바로 뒤 칸부터 foresight 와 '겹쳐' 쓴다 →
        spatial 을 켜도 state·foresight·action 의 위치는 off 일 때와 비트 단위로 같다 (2b 의 블록 위치 공유와 같은 원리)."""
        n_sp = self._n_spatial()
        if n_sp == 0:
            return torch.arange(1, suffix_len + 1)
        n_st = self._n_state_tokens()
        base = torch.arange(1, suffix_len - n_sp + 1)            # off 일 때와 동일한 위치열
        sp = torch.arange(n_st + 1, n_st + 1 + n_sp)            # state 다음 칸부터 n_sp 개 (foresight 위치와 겹침)
        return torch.cat([base[:n_st], sp, base[n_st:]])

    def _isolate_spatial(self, att_2d: torch.Tensor, row_off: int, col_off: int) -> torch.Tensor:
        """foresight ↔ spatial 상호 attention 차단. att_2d: [B, Q, K] bool. row_off/col_off = suffix 시작 행/열.
        action 은 cumsum 규칙으로 spatial·foresight 를 모두 보고, spatial 은 prefix 와 자기 자신만 본다."""
        n_sp = self._n_spatial()
        if n_sp == 0 or not getattr(self.config, "spatial_isolate", True):
            return att_2d
        n_st = self._n_state_tokens()
        n_fs = self.config.num_learnable_tokens
        sp_r = slice(row_off + n_st, row_off + n_st + n_sp)
        fs_r = slice(row_off + n_st + n_sp, row_off + n_st + n_sp + n_fs)
        sp_c = slice(col_off + n_st, col_off + n_st + n_sp)
        fs_c = slice(col_off + n_st + n_sp, col_off + n_st + n_sp + n_fs)
        att_2d = att_2d.clone()
        att_2d[:, fs_r, sp_c] = False
        att_2d[:, sp_r, fs_c] = False
        return att_2d

    def get_spatial_token_output(self, suffix_out: torch.Tensor) -> torch.Tensor:
        n_st = self._n_state_tokens()
        return suffix_out[:, n_st:n_st + self._n_spatial()]

    def _get_spatial_teacher(self):
        """얼린 교사를 첫 학습 forward 에서 지연 생성 (추론·평가 때는 로드되지 않음)."""
        if self._spatial_teacher:
            return self._spatial_teacher[0]
        dev = getattr(self.config, "spatial_teacher_device", "") or self.spatial_tokens.device
        if getattr(self.config, "spatial_teacher", "vggt") == "vggt":
            from vggt.models.vggt import VGGT
            m = VGGT.from_pretrained(self.config.spatial_vggt_name)
            keep = {"point_head"} if getattr(self.config, "spatial_vggt_target", "tokens") == "points" else set()
            for h in ("camera_head", "depth_head", "point_head", "track_head"):   # aggregator (+ points 면 point_head) 만 쓴다
                if hasattr(m, h) and h not in keep:
                    setattr(m, h, None)
            m = m.to(device=dev, dtype=torch.bfloat16).eval()
            if getattr(m, "point_head", None) is not None:
                m.point_head.float()   # DPT 헤드(conv_transpose·LayerNorm)는 fp32 로 — 원본 VGGT 도 헤드는 autocast 밖에서 fp32 실행
            for p in m.parameters():
                p.requires_grad = False
            logging.info(f"[SPATIAL] VGGT teacher loaded on {dev} "
                         f"({sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params, aggregator only)")
            self._spatial_teacher.append(m)
        else:
            self._spatial_teacher.append(self.lf_dino)
        return self._spatial_teacher[0]

    def _select_spatial_views(self, cur: torch.Tensor, views, view_mask):
        """교사 입력 시점 선택 → [B, V, C, H, W] in [-1,1]. V=1 이면 기존(정면 한 장)과 동일.
        'all' 은 배치 내 모든 샘플에 실제로 존재하는 카메라만 쓴다(더미 ones 카메라 제외). image0 은 항상 포함."""
        mode = getattr(self.config, "spatial_views", "image0")
        if mode != "all" or views is None:
            if mode == "all" and not getattr(self, "_sp_views_warned", False):
                logging.warning("[SPATIAL] spatial_views='all' 이지만 배치에 observation.spatial_views 가 없음 → image0 로 폴백 "
                                "(--dataset.spatial_views=true 필요)")
                self._sp_views_warned = True
            return cur[:, None]
        valid = view_mask.all(dim=0) if view_mask is not None else torch.ones(views.shape[1], dtype=torch.bool, device=views.device)
        valid = valid.clone(); valid[0] = True
        if not getattr(self, "_sp_views_logged", False):
            logging.info(f"[SPATIAL] 교사 입력 시점 = {valid.nonzero().flatten().tolist()} (image0 = 정면/머리, 1·2 = 손목)")
            self._sp_views_logged = True
        return views[:, valid]

    @torch.no_grad()
    def _spatial_target(self, cur: torch.Tensor, views: torch.Tensor | None = None,
                        view_mask: torch.Tensor | None = None) -> torch.Tensor:
        """현재 프레임 [B,C,H,W] in [-1,1] (+ 선택: 다중 시점 [B,3,C,H,W]) → 교사 패치 특징을 시점별 g×g 로 평균 풀링한 [B, n_sp, D]."""
        imgs = self._select_spatial_views(cur, views, view_mask)                        # [B, V, C, H, W]
        B, V = imgs.shape[:2]
        img = (imgs.reshape(B * V, *imgs.shape[2:]).to(torch.float32) + 1.0) * 0.5     # [B*V, C, H, W] in [0,1]
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = img.reshape(B, V, *img.shape[1:])
        teacher = self._get_spatial_teacher()
        if getattr(self.config, "spatial_teacher", "vggt") == "vggt":
            tdev = next(teacher.parameters()).device
            outs, ps = teacher.aggregator(img.to(device=tdev, dtype=torch.bfloat16))    # [B,V,3,H,W] (내부 ImageNet 정규화; V>1 이면 시점 간 global attention)
            layer = int(getattr(self.config, "spatial_vggt_layer", -1))
            if layer < 0:
                tok = [o for o in outs if o is not None][-1]
            else:
                if layer >= len(outs) or outs[layer] is None:
                    raise ValueError(f"spatial_vggt_layer={layer} 는 캐시되지 않은 층 (가능: {[i for i, o in enumerate(outs) if o is not None]})")
                tok = outs[layer]
            if getattr(self.config, "spatial_vggt_target", "tokens") == "points":
                # 포인트 헤드: 시점 0 카메라 좌표계의 3D 점 지도 [B,V,H,W,3] (+ 신뢰도). 다중 시점이면 모든 시점이 같은 좌표계.
                outs_f = [o.float() if o is not None else None for o in outs]
                pts, _conf = teacher.point_head(outs_f, images=img.to(device=tdev, dtype=torch.float32), patch_start_idx=ps)
                pts = pts.float().to(cur.device)                                        # [B, V, H, W, 3]
                Bp, Vp, Hp, Wp, _ = pts.shape
                feat = pts.reshape(Bp, Vp, Hp * Wp, 3)                                  # [B, V, H*W, 3] (아래 공통 풀링 경로 사용)
            else:
                feat = tok[:, :, ps:, :].float().to(cur.device)                         # [B, V, 256, 2048] → 모델 장치로
        else:
            x = (img.reshape(B * V, *img.shape[2:]) - self._lf_dino_mean) / self._lf_dino_std
            feat = teacher(pixel_values=x.to(next(teacher.parameters()).dtype)).last_hidden_state[:, 1:].float()
            feat = feat.reshape(B, V, *feat.shape[1:])
        # 타깃 시점: image0 만(기존) 또는 전부. 전부면 토큰 수 = V × g²
        if getattr(self.config, "spatial_target_views", "image0") != "all":
            feat = feat[:, :1]
        Vt = feat.shape[1]
        n_sp = self._n_spatial()
        if n_sp % Vt != 0 or int(round((n_sp // Vt) ** 0.5)) ** 2 != n_sp // Vt:
            raise ValueError(f"spatial_num_tokens={n_sp} 는 타깃 시점 수 {Vt} × 정사각 격자 꼴이어야 함 (예: {Vt}×64={Vt*64})")
        g = int(round((n_sp // Vt) ** 0.5))
        _, _, N, D = feat.shape
        s = int(round(N ** 0.5))
        grid = feat.reshape(B * Vt, N, D).transpose(1, 2).reshape(B * Vt, D, s, s)
        target = F.adaptive_avg_pool2d(grid, (g, g)).flatten(2).transpose(1, 2)        # [B*Vt, g*g, D]
        is_points = getattr(self.config, "spatial_teacher", "vggt") == "vggt" and getattr(self.config, "spatial_vggt_target", "tokens") == "points"
        if is_points:
            if getattr(self.config, "spatial_points_normalize", True):
                # 장면 스케일 제거: 이미지별 평균 거리(원점 = 시점 0 카메라)로 나눔. 다중 시점은 같은 좌표계라 샘플 단위로 나눔.
                scale = target.reshape(B, Vt * g * g, D).norm(dim=-1).mean(dim=1).clamp_min(1e-6)   # [B]
                target = target / scale.repeat_interleave(Vt)[:, None, None]
        elif getattr(self.config, "spatial_target_center", False):
            target = target - target.mean(dim=1, keepdim=True)                           # 시점·이미지별 공통 방향 제거 → 패치 간 차이만 남김
        return target.reshape(B, Vt * g * g, D)                                          # 시점 순서대로 이어 붙임 [B, n_sp, D]

    def _compute_spatial_loss(self, cur_frames: torch.Tensor, spatial_out: torch.Tensor,
                              views: torch.Tensor | None = None, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        pred = self.spatial_decoder(spatial_out).float()                               # [B, n_sp, D]
        target = self._spatial_target(cur_frames, views, view_mask)
        if getattr(self.config, "spatial_loss", "cosine") == "cosine":
            p = F.normalize(pred, dim=-1); t = F.normalize(target, dim=-1)
            return (1.0 - (p * t).sum(dim=-1)).mean()
        return F.mse_loss(pred, target)

    def _spatial_weight_now(self) -> float:
        """손실 가중 × 선형 warm-up (학습 forward 호출 수 기준)."""
        w = float(getattr(self.config, "spatial_loss_weight", 0.0))
        warm = int(getattr(self.config, "spatial_warmup_steps", 0))
        if self.training:
            self._spatial_calls += 1
        if warm <= 0:
            return w
        return w * min(1.0, self._spatial_calls / warm)

    def get_learnable_token_output(self, suffix_out):
        # 상류 슬라이스(start=1) 그대로 유지 — notes_upstream_slice_offbyone.md 참조.
        # spatial 토큰이 앞에 끼면 그 개수만큼 밀어, off 일 때와 같은 토큰(foresight 1..N-1 + action 0)을 본다.
        start = 1 + self._n_spatial()  # skip state token
        end = start + self.config.num_learnable_tokens
        return suffix_out[:, start:end]

    # ------------------------------------------------------------------
    # Inference: denoise step with full suffix output
    # ------------------------------------------------------------------

    def denoise_step_full(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids,
        x_t,
        timestep,
        fast_mask: Tensor | None = None,
    ):
        """Like denoise_step but returns (velocity, learnable_token_output)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        if fast_mask is not None:
            mask_b = fast_mask.to(device=full_att_2d_masks.device, dtype=torch.bool)
            full_att_2d_masks[:, :, :prefix_len] &= ~mask_b[:, None, :]

        full_att_2d_masks = self._isolate_spatial(full_att_2d_masks, row_off=0, col_off=prefix_len)
        position_ids = (
            self._suffix_position_ids_1d(suffix_len).repeat(3, 1, 1).to(max_prefix_position_ids)
            + max_prefix_position_ids
        )

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_5_with_expert.action_expert.config._attn_implementation = "eager"

        outputs_embeds, _ = self.qwen3_5_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        suffix_out = outputs_embeds[1]
        learnable_out = self.get_learnable_token_output(suffix_out)
        action_out = suffix_out[:, -self.config.chunk_size:]
        action_out = action_out.to(dtype=torch.float32)
        velocity = self.action_out_proj(action_out)
        velocity, _ = self._apply_residual_prior(velocity, suffix_out, timestep, state=state)
        return velocity, learnable_out

    # ------------------------------------------------------------------
    # Inference: video generation from learnable tokens
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_video(
        self,
        learnable_out: Tensor,
        cond_frame: Tensor,
        num_inference_steps: int = 50,
    ) -> Tensor:
        """Generate future video frames from learnable token outputs.

        Args:
            learnable_out: [B, N, hidden] learnable token hidden states
            cond_frame: [B, C, 1, H, W] first frame pixels in [-1, 1]
            num_inference_steps: number of denoising steps

        Returns:
            [B, T, C, H, W] generated video pixels in [-1, 1]
        """
        wan_device = next(self.wan_video_model.wan_model.parameters()).device
        wan_dtype = self.wan_video_model.precision

        proj_dtype = self.learnable_to_wan_proj.weight.dtype
        wan_context = self.learnable_to_wan_proj(learnable_out.to(proj_dtype))
        wan_context = wan_context.to(dtype=wan_dtype, device=wan_device)

        cond_frame = cond_frame.to(dtype=wan_dtype, device=wan_device)
        with torch.no_grad():
            cond_latent = self.wan_video_model.encode_video(cond_frame)

        # Ensure batch dimension exists
        if cond_latent.dim() == 4:
            cond_latent = cond_latent.unsqueeze(0)

        B = learnable_out.shape[0]
        C_lat = cond_latent.shape[1]
        lat_H = cond_latent.shape[3]
        lat_W = cond_latent.shape[4]
        lat_T = 1 + self.config.num_video_frames // 4
        latent = torch.randn(B, C_lat, lat_T, lat_H, lat_W, device=wan_device, dtype=wan_dtype)
        latent[:, :, 0:1] = cond_latent

        scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000,
        )
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            t_tensor = t.to(device=wan_device, dtype=wan_dtype)
            with torch.amp.autocast("cuda", dtype=wan_dtype):
                velocity = self.wan_dit_forward(latent, wan_context, t_tensor.expand(B))
            velocity[:, :, 0:1] = 0
            latent = scheduler.step(velocity, t_tensor, latent)
            latent[:, :, 0:1] = cond_latent

        video_pixels = self.wan_video_model.decode_video(latent)
        return video_pixels.permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        actions,
        labels: Tensor | None = None,
        fast_token_mask: Tensor | None = None,
        video_frames: Tensor | None = None,
        video_mask: Tensor | None = None,
        noise=None,
        time=None,
        spatial_views: Tensor | None = None,       # 10.07 [B,3,C,H,W] 모든 카메라 현재 프레임 (다중 시점 교사용, 없으면 None)
        spatial_view_mask: Tensor | None = None,   # 10.07 [B,3] 실제 카메라 여부
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Training forward: action loss + VLM loss + video loss.

        Args:
            video_mask: [B] bool tensor, True for samples with real video frames.
        """
        # WAN(frozen teacher)을 지정 GPU로 1회 이동 (학습 본체 .to() 이후 첫 forward에서).
        # 주의: self.vae(Wan2_2_VAE)는 nn.Module 래퍼가 아니라 .model을 감싸므로 명시적으로 이동.
        _wan_dev = getattr(self, "_wan_target_device", None)
        if _wan_dev is not None and hasattr(self, "wan_video_model"):
            if str(next(self.wan_video_model.wan_model.parameters()).device) != _wan_dev:
                wv = self.wan_video_model
                wv.wan_model.to(_wan_dev)
                if hasattr(wv, "vae") and wv.vae is not None:
                    if getattr(wv.vae, "model", None) is not None:
                        wv.vae.model.to(_wan_dev)
                    # VAE scale=[mean, 1/std] (정규화 상수)도 래퍼 속성이라 별도 이동.
                    _sc = getattr(wv.vae, "scale", None)
                    if isinstance(_sc, (list, tuple)):
                        wv.vae.scale = [s.to(_wan_dev) if torch.is_tensor(s) else s for s in _sc]
                    elif torch.is_tensor(_sc):
                        wv.vae.scale = _sc.to(_wan_dev)
                    if hasattr(wv.vae, "device"):
                        wv.vae.device = torch.device(_wan_dev)
                if hasattr(wv, "device"):
                    wv.device = torch.device(_wan_dev)
                logging.info(f"[WAN-PLACE] WAN(teacher) moved to {_wan_dev}")

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ---- 제안 2b: 다중 표본 (K>1 일 때만 분기; K=1 이면 아래 리스트가 원본 단일 표본 그대로) ----
        k_blocks = max(int(getattr(self.config, "fm_samples", 1)), 1)
        x_t_list, time_list, u_t_list = [x_t], [time], [u_t]
        if k_blocks > 1:
            anti = getattr(self.config, "fm_antithetic", "none")
            t_lo = self.config.time_sampling_offset
            t_hi = t_lo + self.config.time_sampling_scale
            for k in range(1, k_blocks):
                if anti in ("time", "time_noise") and k % 2 == 1:
                    # 대조쌍: t 를 구간 중심에 대해 반사(t → lo+hi−t). 짝수 k 는 다시 새로 샘플링.
                    t_k = (t_lo + t_hi) - time_list[k - 1]
                    # ε 공유 → 두 표본의 타깃 u=ε−a 가 동일해지고, TPC 일관성 항이
                    # "같은 타깃을 향한" 분산 감소 정규화로 성립한다.
                    e_k = -noise if anti == "time_noise" else noise
                else:
                    t_k = self.sample_time(actions.shape[0], actions.device)
                    e_k = self.sample_noise(actions.shape, actions.device)
                te = t_k[:, None, None]
                x_t_list.append(te * e_k + (1 - te) * actions)
                time_list.append(t_k)
                u_t_list.append(e_k - actions)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, labels
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_multi(
            state, x_t_list, time_list
        )

        if (
            self.qwen3_5_with_expert.qwen3_5.language_model.layers[0].mlp.up_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        att_2d_masks = self._isolate_spatial(
            att_2d_masks, row_off=prefix_pad_masks.shape[1], col_off=prefix_pad_masks.shape[1]
        )

        if self.config.block_action_attend_fast_tokens:
            fast_mask = self._compute_fast_token_mask(lang_tokens, fast_token_mask)
            att_2d_masks = self._block_suffix_attend_prefix_tokens(
                att_2d_masks=att_2d_masks,
                prefix_len=prefix_pad_masks.shape[1],
                blocked_prefix_mask=fast_mask,
            )

        # 제안 2b: K개 action 블록을 서로 격리 (각 블록은 prefix·state·foresight 만 참조)
        if k_blocks > 1:
            first_action_idx = (
                prefix_pad_masks.shape[1] + suffix_pad_masks.shape[1] - k_blocks * self.config.chunk_size
            )
            att_2d_masks = self._isolate_action_blocks(
                att_2d_masks, first_action_idx, self.config.chunk_size, k_blocks
            )

        prefix_position_ids, rope_deltas = self.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )

        B = lang_tokens.shape[0]
        suffix_len = suffix_pad_masks.shape[1]

        if labels is not None:
            fast_mask = self._compute_fast_token_mask(lang_tokens, fast_token_mask)
            has_fast = fast_mask.any(dim=1)  # [B]
            first_fast_idx = fast_mask.long().argmax(dim=1)
            anchor_idx = (first_fast_idx - 1).clamp(min=0)

            batch_idx = torch.arange(B, device=lang_tokens.device)
            max_input_pos_before_fast = prefix_position_ids[:, batch_idx, anchor_idx].unsqueeze(-1)
            max_input_pos_all = prefix_position_ids.max(dim=-1, keepdim=True).values

            max_input_pos = torch.where(
                has_fast.view(1, -1, 1),
                max_input_pos_before_fast,
                max_input_pos_all,
            )
        else:
            max_input_pos = prefix_position_ids.max(dim=-1, keepdim=True).values

        if k_blocks > 1:
            # 블록 k 의 h 번째 토큰은 블록 0 의 h 번째와 같은 rope 위치를 갖는다 →
            # 블록 0 의 입력·마스크·위치가 K=1 경로와 완전히 동일해진다.
            chunk = self.config.chunk_size
            head_len = suffix_len - k_blocks * chunk
            n_sp = self._n_spatial()
            head_ids = self._suffix_position_ids_1d(head_len + chunk)[:head_len]   # spatial 포함 head 위치 (off 면 1..head_len)
            blk_ids = torch.arange(head_len - n_sp + 1, head_len - n_sp + chunk + 1)
            suffix_idx = torch.cat([head_ids] + [blk_ids] * k_blocks)
        else:
            suffix_idx = self._suffix_position_ids_1d(suffix_len)
        suffix_position_ids = suffix_idx.repeat(3, 1, 1).to(max_input_pos) + max_input_pos
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (prefix_out, suffix_out), _ = self.qwen3_5_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                knowledge_insulation=self.config.knowledge_insulation,
                use_sdpa=self.config.use_sdpa,
                linear_attn_mask=pad_masks,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids
        )

        # VQA loss (per-token for detailed logging)
        if labels is not None:
            logits = self.qwen3_5_with_expert.qwen3_5.lm_head(prefix_out).to(dtype=torch.float32)
            labels = labels.to(device=logits.device, dtype=torch.long)
            bsize, seq_len, vocab_size = logits.shape
            logits_ar = logits[:, :-1, :].reshape(-1, vocab_size)
            labels_ar = labels[:, 1:].reshape(-1)
            loss_per_token = F.cross_entropy(
                logits_ar, labels_ar, reduction="none", ignore_index=-100
            ).reshape(bsize, seq_len - 1)
            token_mask = (labels[:, 1:] != -100)
            valid = token_mask.float()
            valid_per_sample = valid.sum(dim=1).clamp(min=1)
            loss_vqa = (loss_per_token * valid).sum(dim=1) / valid_per_sample
        else:
            bsize = actions.shape[0]
            loss_per_token = torch.zeros(bsize, 1, device=actions.device, dtype=torch.float32)
            token_mask = torch.zeros(bsize, 1, device=actions.device, dtype=torch.bool)
            loss_vqa = torch.zeros(bsize, device=actions.device, dtype=torch.float32)

        prior_loss = torch.tensor(0.0, device=actions.device)
        tpc_loss = torch.tensor(0.0, device=actions.device)
        # Action loss
        if self.config.video_loss_only:
            loss_action = torch.zeros_like(u_t)
        else:
            chunk = self.config.chunk_size
            v_list = []
            for k in range(k_blocks):
                # 블록 k 의 출력 구간. k_blocks=1 이면 suffix_out[:, -chunk:] 과 동일.
                end = suffix_out.shape[1] - (k_blocks - 1 - k) * chunk
                action_out = suffix_out[:, end - chunk:end].to(dtype=torch.float32)
                v_k = self._apply_checkpoint(lambda x: self.action_out_proj(x), action_out)
                v_k, prior_term = self._apply_residual_prior(
                    v_k, suffix_out, time_list[k], state=state
                )  # off 면 그대로 통과
                v_list.append(v_k)
                # anchored 모드: prior 가 coarse 행동을 직접 맞추도록 지도 (padding 차원은 0 을 학습)
                if prior_term is not None and getattr(self.config, "residual_prior_mode", "velocity") == "anchored":
                    prior_loss = prior_loss + F.mse_loss(
                        prior_term.float(), actions.float(), reduction="mean"
                    ) / k_blocks

            # 제안 1·3 이 꺼져 있으면 _shape_action_loss 는 (v−u)² = F.mse_loss(reduction="none") 과 동일.
            losses_k = [
                self._shape_action_loss(v_list[k], u_t_list[k], actions, time_list[k])
                for k in range(k_blocks)
            ]
            loss_action = losses_k[0] if k_blocks == 1 else torch.stack(losses_k, 0).mean(0)

            # TPC 일관성 항: ε 를 공유하는 대조쌍에서만 의미가 있다(타깃 u 가 동일).
            tpc_w = float(getattr(self.config, "fm_tpc_weight", 0.0))
            if k_blocks >= 2 and tpc_w > 0.0:
                if getattr(self.config, "fm_antithetic", "none") == "none":
                    raise ValueError("fm_tpc_weight>0 은 fm_antithetic=time|time_noise 에서만 유효")
                pairs = [(k, k + 1) for k in range(0, k_blocks - 1, 2)]
                gate_p = float(getattr(self.config, "fm_tpc_prob", 0.75))
                acc = torch.zeros((), device=actions.device, dtype=torch.float32)
                for a, b in pairs:
                    diff = (v_list[a] - v_list[b]).pow(2).mean(dim=(1, 2))  # [B]
                    gate = (torch.rand_like(diff) < gate_p).float()         # 과도한 제약 방지(Bernoulli 게이팅)
                    acc = acc + (diff * gate).mean()
                tpc_loss = tpc_w * acc / len(pairs)

        video_frames_all = video_frames   # spatial 손실용 (아래 video 블록이 video_frames 를 마스킹·재할당함)
        # Video loss — only computed for samples with real video frames.
        # (video_loss 슬롯: WAN 픽셀 loss / Latent Foresight 회귀 loss / 0 중 하나)
        use_lf = getattr(self.config, "use_latent_foresight", False)
        if self.config.action_loss_only:
            video_loss = torch.tensor(0.0, device=actions.device)
        else:
            has_video = video_mask.any() if video_mask is not None else (video_frames is not None)
            if has_video:
                learnable_out = self.get_learnable_token_output(suffix_out).to(dtype=torch.float32)
                if video_mask is not None:
                    video_frames = video_frames[video_mask]
                    learnable_out = learnable_out[video_mask]
                if use_lf:
                    # Latent Foresight: WAN 없이 미래 프레임의 coarse 표현으로 회귀.
                    video_loss = self._compute_latent_foresight_loss(video_frames, learnable_out).to(actions.device)
                else:
                    # WAN이 다른 GPU(wan_device)에 있을 수 있으므로 결합 전 action device로 이동.
                    video_loss = self._compute_video_loss(video_frames, learnable_out).to(actions.device)
            else:
                video_loss = torch.tensor(0.0, device=actions.device)

        # Spatial query tokens (9.29): 현재 프레임(video_frames[:,0])의 얼린 교사 특징 회귀 — 로봇 샘플만
        spatial_loss = torch.tensor(0.0, device=actions.device)
        if self._n_spatial() > 0 and video_frames_all is not None and (video_mask is None or video_mask.any()):
            sp_out = self.get_spatial_token_output(suffix_out).to(dtype=torch.float32)
            cur = video_frames_all[:, 0]
            sv, svm = spatial_views, spatial_view_mask
            if video_mask is not None:
                sp_out = sp_out[video_mask]
                cur = cur[video_mask]
                if sv is not None:
                    sv = sv[video_mask]
                    svm = svm[video_mask] if svm is not None else None
            spatial_loss = self._spatial_weight_now() * self._compute_spatial_loss(cur, sp_out, sv, svm).to(actions.device)

        return loss_action, loss_vqa, video_loss, loss_per_token, token_mask, prior_loss, tpc_loss, spatial_loss

    # ------------------------------------------------------------------
    # WAN DiT forward (cross-attention conditioning)
    # ------------------------------------------------------------------

    def wan_dit_forward(
        self,
        noisy_video_latent: torch.Tensor,
        wan_context: torch.Tensor,
        video_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run WAN DiT with learnable token context as cross-attention K/V."""
        wan = self.wan_video_model.wan_model
        device = wan.patch_embedding.weight.device
        if wan.freqs.device != device:
            wan.freqs = wan.freqs.to(device)

        B = noisy_video_latent.shape[0]

        # Patch embedding
        x = wan.patch_embedding(noisy_video_latent)
        grid_sizes = self._wan_grid_sizes.unsqueeze(0).expand(B, -1).to(device)
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.shape[1]
        seq_lens = torch.full((B,), seq_len, dtype=torch.long, device=device)

        # Time embedding
        t_vid = video_timestep
        if t_vid.dim() == 1:
            t_vid = t_vid.unsqueeze(1).expand(B, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, t_vid.flatten())
                .unflatten(0, (B, seq_len))
                .float()
                .to(device)
            )
            e0 = wan.time_projection(e).unflatten(2, (6, wan.dim))

        # Use projected learnable tokens as context (skip text_embedding)
        context = wan_context

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=wan.freqs,
            context=context,
            context_lens=None,
        )

        for block in wan.blocks:
            x = block(x, **kwargs)

        x = wan.head(x, e)
        x = wan.unpatchify(x, grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    # ------------------------------------------------------------------
    # Video loss computation
    # ------------------------------------------------------------------

    def _compute_latent_foresight_loss(
        self,
        video_frames: torch.Tensor,
        learnable_out: torch.Tensor,
    ) -> torch.Tensor:
        """Latent Foresight loss (Lumo-2 이식): 얼린 WAN 대신 경량 회귀.

        foresight 토큰(N개)을 mean-pool → head → 미래 프레임의 coarse 표현(3·g·g)을
        예측하고 MSE. WAN diffusion(5B) 없이 dynamics를 grounding. 추론 시 foresight는
        폐기되므로 추론 성능 손실 0.

        Args:
            video_frames: [B, T, C, H, W] in [-1, 1] (index 0=현재 조건 프레임, 마지막=미래)
            learnable_out: [B, N, hidden] extracted learnable token outputs (float32)
        """
        # 미래 = 시퀀스 마지막 프레임 (가장 먼 horizon → foresight 신호 최대)
        future = video_frames[:, -1].to(dtype=torch.float32)          # [B, C, H, W] in [-1,1]
        if getattr(self.config, "latent_foresight_target", "pixel") == "dino":
            return self._lf_dino_loss(future, learnable_out)          # v2
        # v1: coarse 픽셀 MSE
        g = self.config.latent_foresight_grid
        target = F.adaptive_avg_pool2d(future, (g, g)).flatten(1)     # [B, 3·g·g]
        pred = self.latent_foresight_head(learnable_out.mean(dim=1))  # [B, 3·g·g]
        return F.mse_loss(pred, target, reduction="mean")

    def _lf_dino_loss(self, future: torch.Tensor, learnable_out: torch.Tensor) -> torch.Tensor:
        """v2: foresight 토큰 → cross-attn 디코더 → 얼린 DINOv2 patch feature 회귀."""
        img = (future + 1.0) * 0.5                                    # [-1,1] → [0,1]
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self._lf_dino_mean) / self._lf_dino_std          # ImageNet 정규화
        with torch.no_grad():
            dino_dtype = next(self.lf_dino.parameters()).dtype
            out = self.lf_dino(pixel_values=img.to(dino_dtype))
            target = out.last_hidden_state[:, 1:].float()            # [B, N, D] (CLS 제외)
        pred = self.latent_foresight_decoder(learnable_out).float()  # [B, N, D]
        loss_type = getattr(self.config, "latent_foresight_loss", "mse")
        if loss_type == "cosine":
            p = F.normalize(pred, dim=-1); t = F.normalize(target, dim=-1)
            return (1.0 - (p * t).sum(dim=-1)).mean()
        if loss_type == "infonce":
            B, N, D = pred.shape
            p = F.normalize(pred.reshape(B * N, D), dim=-1)
            t = F.normalize(target.reshape(B * N, D), dim=-1)
            logits = (p @ t.t()) / 0.1                               # [BN, BN]
            labels = torch.arange(B * N, device=p.device)
            return F.cross_entropy(logits, labels)
        return F.mse_loss(pred, target, reduction="mean")

    def _compute_video_loss(
        self,
        video_frames: torch.Tensor,
        learnable_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WAN video prediction loss via flow matching.

        Args:
            video_frames: [B, T, C, H, W] in [-1, 1]
            learnable_out: [B, N, hidden] extracted learnable token outputs
        """
        B = video_frames.shape[0]
        wan_device = next(self.wan_video_model.wan_model.parameters()).device
        wan_dtype = self.wan_video_model.precision

        # Project learnable tokens to WAN context
        wan_context = self.learnable_to_wan_proj(learnable_out)
        wan_context = wan_context.to(dtype=wan_dtype, device=wan_device)

        # Encode video with frozen VAE: expects [B, C, T, H, W]
        video_bcthw = video_frames.permute(0, 2, 1, 3, 4).to(dtype=wan_dtype, device=wan_device)
        first_frame_bcthw = video_bcthw[:, :, 0:1]

        with torch.no_grad():
            clean_latent = self.wan_video_model.encode_video(video_bcthw)
            cond_latent = self.wan_video_model.encode_video(first_frame_bcthw)

        # Sample video flow-matching timestep
        timestep_id = torch.randint(
            0, self.fm_video_scheduler.num_train_timesteps, (B,)
        )
        sigma = self.fm_video_scheduler.sigmas[timestep_id].to(
            dtype=wan_dtype, device=wan_device
        ).view(B, 1, 1, 1, 1)
        video_t = self.fm_video_scheduler.timesteps[timestep_id].to(
            dtype=wan_dtype, device=wan_device
        )

        # Add noise (teacher forcing: keep frame 0 clean)
        video_noise = torch.randn_like(clean_latent)
        noisy_latent = clean_latent * (1 - sigma) + video_noise * sigma
        noisy_latent[:, :, 0:1] = cond_latent

        # Target velocity
        video_target = video_noise - clean_latent
        video_target[:, :, 0:1] = 0

        # WAN forward
        with torch.amp.autocast("cuda", dtype=wan_dtype):
            video_pred = self.wan_dit_forward(noisy_latent, wan_context, video_t)

        video_pred[:, :, 0:1] = 0
        return F.mse_loss(video_pred.float(), video_target.float(), reduction="mean")


# ======================================================================
# Policy wrapper
# ======================================================================

class InternVLAA15Policy(PreTrainedPolicy):
    """Qwen3.5 VLA policy with WAN video auxiliary supervision."""

    config_class = InternVLAA15Config
    name = "internvla_a1_5"
    _checkpoint_excluded_prefixes = ("model.wan_video_model.",)

    def __init__(self, config: InternVLAA15Config):
        super().__init__(config)
        config.validate_features()
        self.config = config

        if config.inference_backend == "optimized":
            from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5_optimized import (
                InternVLAA15Optimized,
            )

            self.model = InternVLAA15Optimized(config)
        else:
            self.model = InternVLAA15(config)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def __str__(self) -> str:
        lines = ["=" * 60, f"Policy: {self.__class__.__name__}", ""]
        num_total = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_vlm = sum(p.numel() for p in self.model.qwen3_5_with_expert.qwen3_5.parameters())
        num_expert = sum(p.numel() for p in self.model.qwen3_5_with_expert.action_expert.parameters())
        num_wan = (
            sum(p.numel() for p in self.model.wan_video_model.parameters())
            if hasattr(self.model, "wan_video_model")
            else 0
        )
        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {format_big_number(num_total)}")
        lines.append(f"  - Trainable params    : {format_big_number(num_trainable)}")
        lines.append(f"  - Qwen3_5 params      : {format_big_number(num_vlm)}")
        lines.append(f"  - Action expert params: {format_big_number(num_expert)}")
        lines.append(f"  - WAN params          : {format_big_number(num_wan)}")
        lines.append(f"  - Learnable tokens    : {self.config.num_learnable_tokens}")
        lines.append(f"  - Knowledge insulation: {self.config.knowledge_insulation}")
        lines.append(f"  - Inference backend   : {self.config.inference_backend}")
        lines.append(f"  - Freeze WAN DiT      : {self.config.freeze_wan_dit}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.action_out_proj.to(torch.float32)
        if hasattr(self.model, "_cast_action_path_to_fp32"):
            self.model._cast_action_path_to_fp32()
        return self

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith(self._checkpoint_excluded_prefixes):
                del state[key]

        metadata = getattr(state, "_metadata", None)
        if metadata is not None:
            for key in list(metadata.keys()):
                if key == "model.wan_video_model" or key.startswith("model.wan_video_model."):
                    del metadata[key]
        return state

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def prepare_state(self, batch):
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        fast_token_mask = batch.get(f"{OBS_PREFIX}fast_token_mask")
        state = self.prepare_state(batch)

        actions = self.model.sample_actions(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
            fast_token_mask=fast_token_mask,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def predict_action_chunk_with_video(
        self, batch: dict[str, Tensor], num_video_steps: int = 50
    ) -> tuple[Tensor, Tensor]:
        """Predict actions and generate future video from learnable tokens.

        Returns:
            (actions [B, chunk, action_dim], video [B, T, C, H, W] in [-1, 1])
        """
        self.eval()
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        fast_token_mask = batch.get(f"{OBS_PREFIX}fast_token_mask")
        state = self.prepare_state(batch)

        model = self.model
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype
        num_steps = model.config.num_inference_steps

        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        noise = model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, _ = model.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.qwen3_5_with_expert.qwen3_5.language_model.config._attn_implementation = "eager"

        _, past_key_values = model.qwen3_5_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            knowledge_insulation=model.config.knowledge_insulation,
        )

        dt = -1.0 / num_steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        if model.config.block_action_attend_fast_tokens:
            fast_mask = model._compute_fast_token_mask(lang_tokens, fast_token_mask)
        else:
            fast_mask = None

        # Denoising loop - use full step on last iteration to capture learnable outputs
        learnable_out = None
        while time >= -dt_tensor / 2:
            expanded_time = time.expand(bsize)
            is_last = (time + dt_tensor) < -dt_tensor / 2
            if is_last:
                v_t, learnable_out = model.denoise_step_full(
                    state, prefix_pad_masks, past_key_values,
                    max_prefix_position_ids, x_t.to(dtype), expanded_time.to(dtype),
                    fast_mask=fast_mask,
                )
            else:
                v_t = model.denoise_step(
                    state, prefix_pad_masks, past_key_values,
                    max_prefix_position_ids, x_t.to(dtype), expanded_time.to(dtype),
                    fast_mask=fast_mask,
                )
            x_t = x_t + dt_tensor * v_t
            time += dt_tensor

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = x_t[:, :, :original_action_dim]

        # Generate video from learnable token outputs
        video_frames = batch.get("observation.video_frames")
        if video_frames is not None:
            cond_frame = video_frames[:, 0:1].permute(0, 2, 1, 3, 4)
        else:
            cond_frame = torch.zeros(
                bsize, 3, 1, model.config.video_height, model.config.video_width,
                device=device, dtype=dtype,
            )

        generated_video = model.generate_video(
            learnable_out, cond_frame, num_inference_steps=num_video_steps
        )
        return actions, generated_video

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        fast_token_mask = batch.get(f"{OBS_PREFIX}fast_token_mask")
        video_frames = batch.get("observation.video_frames")

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        labels = batch["VQA.labels"] if self.config.enable_vqa_loss else None

        # video_mask: True for robot samples that have real video frames
        vqa_type = batch.get("vqa_type")
        if vqa_type is not None:
            video_mask = (vqa_type != 1)  # VQA-only samples have vqa_type=1
        else:
            video_mask = None

        losses, losses_vlm, video_loss, loss_per_token, token_mask, prior_loss, tpc_loss, spatial_loss = self.model.forward(
            pixel_values, image_grid_thw, lang_tokens, lang_masks,
            state, actions,
            labels=labels,
            fast_token_mask=fast_token_mask,
            video_frames=video_frames,
            video_mask=video_mask,
            spatial_views=batch.get("observation.spatial_views"),            # 10.07 (dataset.spatial_views=true 일 때만 존재)
            spatial_view_mask=batch.get("observation.spatial_view_mask"),
        )

        original_action_dim = batch[ACTION].shape[-1]
        losses = losses[:, :, :original_action_dim]
        zero = torch.tensor(0.0, device=losses.device)

        # =====================================================================
        # Loss taxonomy:
        #
        # [VLM branch] prefix → lm_head → next-token cross-entropy
        #   loss_vlm       : mean CE over all valid label tokens
        #   loss_fast      : CE on fast action tokens (token_id ∈ [action_token_min, max])
        #   loss_subtask   : CE on subtask/text tokens (valid but not fast)
        #
        # [Action Expert branch] suffix → action_out_proj → flow matching MSE
        #   loss_fm_action : mean MSE of predicted velocity vs target
        #
        # [Video branch] suffix learnable tokens → WAN DiT → MSE
        #   loss_video     : video prediction flow matching loss
        # =====================================================================

        # --- VLM branch: per-token split ---
        if labels is not None and token_mask.any():
            labels_shifted = labels[:, 1:]
            fast_tok_mask = (
                (labels_shifted >= self.config.action_token_min)
                & (labels_shifted <= self.config.action_token_max)
            )
            subtask_tok_mask = token_mask & ~fast_tok_mask

            fast_sum = (loss_per_token * fast_tok_mask.float()).sum()
            fast_cnt = fast_tok_mask.float().sum()
            subtask_sum = (loss_per_token * subtask_tok_mask.float()).sum()
            subtask_cnt = subtask_tok_mask.float().sum()

            loss_fast = (fast_sum / fast_cnt) if fast_cnt > 0 else zero
            loss_subtask = (subtask_sum / subtask_cnt) if subtask_cnt > 0 else zero
        else:
            loss_fast = zero
            loss_subtask = zero

        # --- Per-sample aggregation by vqa_type ---
        if self.config.enable_vqa_loss:
            vqa_type = batch["vqa_type"]
            action_mask = (vqa_type == 0) | (vqa_type == 2)  # robot samples
            vlm_mask = (vqa_type == 1) | (vqa_type == 2)     # samples with VQA labels

            loss_fm_action = losses[action_mask].mean() if action_mask.any() else zero
            loss_vlm = losses_vlm[vlm_mask].mean() if vlm_mask.any() else zero

            loss = (
                10 * loss_fm_action
                # loss_fm_action
                + self.config.lambda_vqa * loss_vlm
                + self.config.video_loss_weight * video_loss
                + getattr(self.config, "residual_prior_loss_weight", 1.0) * prior_loss
                + tpc_loss  # 가중치는 fm_tpc_weight 에서 이미 곱해짐 (off 면 0)
                + spatial_loss  # 9.29 기하 교사 손실 — 가중·warm-up 은 모델 안에서 곱해짐 (off 면 0)
            )
            # --- Build loss_dict ---
            loss_dict = {
                "loss": loss.item(),
                # VLM branch
                "loss_vqa": loss_vlm.item(),
                "loss_fast": loss_fast.item(),
                "loss_subtask": loss_subtask.item(),
                # Action Expert branch
                "loss_action": loss_fm_action.item(),
                # Video branch
                "loss_video": video_loss.item(),
                "loss_prior": float(prior_loss),
                "loss_tpc": float(tpc_loss),
                "loss_spatial": float(spatial_loss),
            }
        else:
            loss_fm_action = losses.mean()
            loss_vlm = zero
            loss = (
                loss_fm_action
                + self.config.video_loss_weight * video_loss
                + getattr(self.config, "residual_prior_loss_weight", 1.0) * prior_loss
                + tpc_loss
                + spatial_loss
            )
            loss_dict = {
                "loss": loss.item(),
                # Action Expert branch
                "loss_action": loss_fm_action.item(),
                # Video branch
                "loss_video": video_loss.item(),
                "loss_tpc": float(tpc_loss),
                "loss_spatial": float(spatial_loss),
            }
        return loss, loss_dict
