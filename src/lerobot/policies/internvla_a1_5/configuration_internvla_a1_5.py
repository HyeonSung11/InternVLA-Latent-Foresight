import logging
from dataclasses import dataclass, field, replace
from typing import Sequence

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    ExtractVideoFramesTransformFn,
    FASTInternVLAA15ActionTokenizerTransformFn,
    InternVLAA15ChatProcessorTransformFn,
    InternVLAA15VQAProcessorTransformFn,
    SplitActionMemoryTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import HF_HOME, OBS_IMAGES


@DatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenize_state: bool = False
    max_prompt_length: int = 650
    mode: str = "train"
    chunk_size: int = 50
    use_fast_action_tokens: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    # Memory ω (Lumo-2 이식 · 기본 off): 과거 액션을 prefix 문자열로 주입
    use_action_memory: bool = False
    action_memory_len: int = 8
    # 10.07 다중 시점 기하 교사(기본 off): 모든 카메라의 현재 프레임을 observation.spatial_views 로 배치에 실어 보냄
    spatial_views: bool = False

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=InternVLAA15DatasetConfig.height,
                    width=InternVLAA15DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                ExtractVideoFramesTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                FASTInternVLAA15ActionTokenizerTransformFn(),
                LoadActionTextFromJsonlTransformFn(),
                InternVLAA15ChatProcessorTransformFn(),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15DatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15DatasetConfig.max_action_dim,
                ),
                ReorderStateActionTransform(),
                UnifyInternVLAA15InputsTransformFn(
                    num_video_frames=InternVLAA15DatasetConfig.num_video_frames,
                    video_height=InternVLAA15DatasetConfig.video_height,
                    video_width=InternVLAA15DatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta" and not has_delta:
            logging.info("action_mode='delta' -> Adding DeltaActionTransformFn")
            inputs = [DeltaActionTransformFn(), *inputs]
        elif self.action_mode == "abs" and has_delta:
            logging.info("action_mode='abs' -> Removing DeltaActionTransformFn")
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]

        processor = InternVLAA15ChatProcessorTransformFn(
            tokenize_state=self.tokenize_state,
            max_state_dim=self.max_state_dim,
            max_length=self.max_prompt_length,
            use_fast_action_tokens=self.use_fast_action_tokens,
            mode=self.mode,
            use_action_memory=self.use_action_memory,
            action_memory_len=self.action_memory_len,
            max_action_dim=self.max_action_dim,
        )
        inputs = [t for t in inputs if not isinstance(t, InternVLAA15ChatProcessorTransformFn)]

        # Memory ω: 과거/미래 액션 분리 transform을 맨 앞에 삽입(정규화·FAST 이전).
        # off면 삽입 안 함 → 파이프라인 원본과 동일.
        inputs = [t for t in inputs if not isinstance(t, SplitActionMemoryTransformFn)]
        if self.use_action_memory:
            inputs.insert(0, SplitActionMemoryTransformFn(action_memory_len=self.action_memory_len))
        insert_idx = next(
            (i for i, t in enumerate(inputs) if isinstance(t, PadStateAndActionTransformFn)),
            len(inputs),
        )
        inputs.insert(insert_idx, processor)

        has_fast = any(isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn) for t in inputs)
        if self.use_fast_action_tokens and not has_fast:
            inputs.insert(insert_idx, FASTInternVLAA15ActionTokenizerTransformFn())
        elif not self.use_fast_action_tokens and has_fast:
            inputs = [t for t in inputs if not isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn)]

        for t in inputs:
            if isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn):
                t.chunk_size = self.chunk_size
                break

        # 10.07 다중 시점 교사: on 일 때만 추출·통과 transform 에 플래그 전달 (off 면 원본과 동일)
        if self.spatial_views:
            inputs = [replace(t, save_views=True)
                      if isinstance(t, (ExtractVideoFramesTransformFn, UnifyInternVLAA15InputsTransformFn)) else t
                      for t in inputs]

        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@DataTransformFn.register_subclass("unify_internvla_a1_5_inputs")
@dataclass
class UnifyInternVLAA15InputsTransformFn(DataTransformFn):
    """Unify robot samples and always include video_frames for WAN.

    Always outputs observation.video_frames so that robot and VQA samples
    have identical keys and can be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    save_views: bool = False   # 10.07 다중 시점 교사: spatial_views / spatial_view_mask 를 배치 키로 통과

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_NONE
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_NONE, dtype=torch.long))

        video_key = "observation.video_frames"
        if video_key in data:
            video_frames = data[video_key]
        else:
            video_frames = torch.zeros(
                self.num_video_frames + 1, 3, self.video_height, self.video_width
            )

        out = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": data["vqa_type"],
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            video_key: video_frames,
        }
        if self.save_views:
            out["observation.spatial_views"] = data.get(
                "observation.spatial_views", torch.zeros(3, 3, self.video_height, self.video_width))
            out["observation.spatial_view_mask"] = data.get(
                "observation.spatial_view_mask", torch.zeros(3, dtype=torch.bool))
        return out


@DataTransformFn.register_subclass("unify_internvla_a1_5_vqa_inputs")
@dataclass
class UnifyInternVLAA15VQAInputsTransformFn(DataTransformFn):
    """VQA unify transform that includes a dummy video_frames tensor.

    Ensures VQA samples have the same keys as robot samples so they can
    be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    save_views: bool = False   # 10.07 다중 시점 교사: 로봇 배치와 키를 맞추기 위한 더미

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_TEXT
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_TEXT, dtype=torch.long))

        video_frames = torch.zeros(
            self.num_video_frames + 1, 3, self.video_height, self.video_width
        )

        out = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": torch.tensor(1, dtype=torch.long),
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            "observation.video_frames": video_frames,
        }
        if self.save_views:
            out["observation.spatial_views"] = torch.zeros(3, 3, self.video_height, self.video_width)
            out["observation.spatial_view_mask"] = torch.zeros(3, dtype=torch.bool)
        return out


@VQADatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15VQADatasetConfig(VQADatasetConfig):
    """VQA dataset config with dummy video_frames for shared batching."""

    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    spatial_views: bool = False   # 10.07 로봇 dataset 의 spatial_views 와 같이 켠다

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # Note: If you resize the VQA images, the coordinates in the VQA Data should be scaled accordingly. 
                # Please pre-process the VQA data offline based on its format.
                ResizeVQAImagesWithPadFn(
                    height=InternVLAA15VQADatasetConfig.height,
                    width=InternVLAA15VQADatasetConfig.width,
                ),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15VQADatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15VQADatasetConfig.max_action_dim,
                ),
                InternVLAA15VQAProcessorTransformFn(),
                UnifyInternVLAA15VQAInputsTransformFn(
                    num_video_frames=InternVLAA15VQADatasetConfig.num_video_frames,
                    video_height=InternVLAA15VQADatasetConfig.video_height,
                    video_width=InternVLAA15VQADatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        inputs = list(self.data_transforms.inputs)
        processor_type = InternVLAA15VQAProcessorTransformFn
        inputs = [t for t in inputs if not isinstance(t, processor_type)]
        insert_idx = next(
            (
                i
                for i, t in enumerate(inputs)
                if isinstance(t, UnifyInternVLAA15VQAInputsTransformFn)
            ),
            len(inputs),
        )
        inputs.insert(insert_idx, InternVLAA15VQAProcessorTransformFn())
        if self.spatial_views:   # 10.07 로봇 배치와 키를 맞추는 더미
            inputs = [replace(t, save_views=True) if isinstance(t, UnifyInternVLAA15VQAInputsTransformFn) else t
                      for t in inputs]
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15Config(PreTrainedConfig):
    # VLM model selection - supports Qwen3.5-2B/4B/8B
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"

    # Action expert customization
    action_expert_hidden_size: int | None = 1024
    action_expert_intermediate_size: int | None = 3072

    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # VQA configurations
    enable_vqa_loss: bool = True
    lambda_vqa: float = 1.0
    tokenize_state: bool = True

    # FAST action tokens
    action_token_min: int = 248077
    action_token_max: int = 250124

    # Knowledge insulation
    knowledge_insulation: bool = False
    block_action_attend_fast_tokens: bool = True

    inference_action_type: str = "fm"  # "fm" for flow matching, "fast" for fast-token supervision
    inference_backend: str = "standard"  # "standard" or "optimized"

    # Use SDPA (scaled_dot_product_attention) instead of eager attention in training
    use_sdpa: bool = False

    num_learnable_tokens: int = 50

    wan_checkpoint_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    wan_config_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    vae_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    video_precision: str = "bfloat16"

    freeze_wan_dit: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    video_loss_weight: float = 1.0
    video_loss_only: bool = False
    action_loss_only: bool = False
    freeze_learnable_tokens: bool = False

    # ---- Seam Grounding (TurboVLA 이식 · 연구 기능 · 기본 off) ----
    # 켜면 prefix 임베딩(H_t 원료)의 시각↔지시 토큰을 얕은 양방향 cross-attn으로
    # 명시 정렬해 grounded prefix를 만든다. off면 원본 동작·파라미터·VRAM 모두 그대로.
    use_seam_grounding: bool = False
    seam_num_layers: int = 4
    seam_num_heads: int = 8
    seam_detach_input: bool = True   # KI: VLM 특징을 detach로 읽어 VLM에 grad 역류 차단
    # ---- VRAM 절감: VLM 백본 동결 (연구 학습 시 seam/expert만 학습) · 기본 off ----
    freeze_vlm: bool = False
    # ---- VLM LoRA (동결 병목 완화 · 연구 기능 · 기본 off) ----
    # base VLM은 동결하고 저랭크 adapter(A,B)만 학습 → 소량 VRAM으로 VLM을 태스크에 적응.
    # B=zero-init → 시작 시 원본과 동일(항등). off면 파라미터·VRAM 0.
    use_vlm_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # 교체 대상 Linear 하위 모듈명 (Qwen3.5 어텐션 투영). MLP까지 원하면 gate/up/down_proj 추가.
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # ---- 8-bit AdamW (bitsandbytes) · full finetune VRAM 절감 · 기본 off ----
    optimizer_8bit: bool = False
    # ---- WAN(frozen teacher)을 별도 GPU에 배치 → 학습 본체 GPU VRAM 확보 · 기본 "" (본체와 동일) ----
    wan_device: str = ""
    # ---- Latent Foresight (Lumo-2 이식 · 연구 기능 · 기본 off) ----
    # 켜면 얼린 WAN(5B 픽셀 diffusion) 대신, foresight 토큰을 미래 프레임의 압축 표현으로
    # 회귀(MSE)하는 경량 head를 학습한다. WAN을 아예 로드하지 않아 학습 VRAM −10GB.
    # 추론 시 foresight는 원래 폐기되므로 추론 성능 손실 0. off면 원본 WAN 경로 그대로.
    use_latent_foresight: bool = False
    latent_foresight_grid: int = 8   # (pixel 타깃) 미래 프레임 coarse 공간 grid (g×g, 채널3)
    # v1(pixel/mlp/mse) ↔ v2(dino/xattn/cosine) 를 한 코드에서 flag로 계층화:
    latent_foresight_target: str = "pixel"          # pixel(v1) | dino(v2, frozen DINOv2 feature)
    latent_foresight_loss: str = "mse"              # mse(v1) | cosine | infonce
    latent_foresight_dino_name: str = "facebook/dinov2-small"  # v2 frozen target 인코더
    latent_foresight_decoder_layers: int = 3
    # ---- Spatial Query Tokens (9.29 · 기하 쿼리 토큰, flag일 때만 생성 → off면 원본과 동일) ----
    use_spatial_tokens: bool = False
    spatial_num_tokens: int = 64            # 8×8 격자
    spatial_teacher: str = "vggt"           # vggt | dino (ablation: 현재 프레임 DINOv2 — LF v2 와 정보 중복 경고)
    spatial_vggt_name: str = "facebook/VGGT-1B"
    spatial_teacher_device: str = ""        # 얼린 교사를 둘 장치 ("" = 모델과 같은 GPU, "cuda:1" = 원본 WAN 처럼 옆 GPU → GPU0 메모리·paged optimizer 압박 회피)
    spatial_loss: str = "cosine"            # cosine | mse
    spatial_vggt_layer: int = -1            # aggregator 출력 층 (-1 = 마지막 23층; 캐시된 층 4·11·17·23 중 선택). 마지막층은 패치 간 공통 방향이 96%
    spatial_target_center: bool = False     # 타깃에서 이미지별 패치 평균을 빼 공통 방향 제거 (상수 예측으로 cosine 0.975 가 나오는 퇴화 방지)
    spatial_loss_weight: float = 0.15
    spatial_warmup_steps: int = 1000        # 손실 가중 선형 warm-up (학습 forward 호출 수 기준)
    spatial_isolate: bool = True            # foresight ↔ spatial 상호 attention 차단
    spatial_decoder_layers: int = 3        # v2 cross-attn 디코더 층수
    # 10.07 다중 시점 교사. "image0" = 정면 한 장(기존) / "all" = 실제 카메라 전부를 VGGT 에 함께 입력 (시점 간 attention 으로 3D 대응)
    #   → batch 에 observation.spatial_views 가 있어야 함 (--dataset.spatial_views=true). 없으면 image0 로 폴백 + 경고 1회.
    spatial_views: str = "image0"
    # 타깃으로 쓸 시점. "image0" = 정면 64 토큰만 / "all" = 시점마다 g×g → spatial_num_tokens = 시점 수 × g² 이어야 함 (예: 3시점 192)
    spatial_target_views: str = "image0"
    # 10.10 VGGT 타깃 종류. "tokens" = aggregator 패치 특징(2048, 기존) / "points" = 포인트 헤드의 3D 점 지도(카메라 좌표계 xyz, 3차원)
    #   points 는 행동과 직결되는 기하("이 칸이 얼마나 멀리·어디에")이고 공통 방향 퇴화가 없음. 손실은 spatial_loss=mse 권장.
    spatial_vggt_target: str = "tokens"
    spatial_points_normalize: bool = True   # 이미지(장면)마다 풀링된 점들의 평균 거리로 나눠 VGGT 의 임의 장면 스케일을 제거
    # ---- 단기 행동 메모리 ω (Lumo-2 이식 · 연구 기능 · 기본 off) ----
    # 켜면 과거 action_memory_len개 액션을 discretize해 "ActionMemory: ..." 문자열로 prefix에
    # 주입 → perceptual aliasing(같은 관측·다른 단계) 해소. action_delta_indices를 과거로 확장.
    # off면 원본 동작·데이터로딩 그대로.
    use_action_memory: bool = False
    action_memory_len: int = 8
    # ---- Foresight Residual Prior (DRIFT 변형 · 연구 기능 · 기본 off) ----
    # 켜면 foresight 토큰 출력(예측된 미래 잠재)을 mean-pool 해 시각 t 와 함께 작은 MLP 에 넣고,
    # 그 출력을 expert 의 velocity 예측에 잔차로 더한다:  v_hat = v_pred + r(foresight, t).
    # x-공간으로 보면 잔차가 t 배로 들어가 고노이즈 구간에서 강하고 t→0 에서 사라지는 time-gated prior.
    # 출력층 zero-init → step 0 에서 원본과 완전히 동일(항등). off면 파라미터·VRAM 0.
    use_residual_prior: bool = False
    residual_prior_source: str = "foresight"   # foresight | state  (ablation 용)
    residual_prior_hidden: int = 0             # 0 → action_expert_hidden_size
    # v1 "velocity": v_hat = v_pred + r (비지도 잔차 · 25k 결과 r≈0 으로 사실상 무효, 09-14 진단)
    # v2 "anchored": prior 가 coarse 행동 g 를 직접 예측(λ·‖g−a‖² 지도)하고 flow 는 잔차만: v_hat = v_pred − g
    residual_prior_mode: str = "velocity"      # velocity | anchored
    residual_prior_loss_weight: float = 1.0    # anchored 모드의 g 지도 손실 가중치

    # ==================================================================
    # action loss 계열 연구 플래그 (09-15) — 전부 파라미터 0개 추가, 기본 off.
    # Residual Prior 3종이 전부 하락한 원인이 "gradient 경로에 새 파라미터 추가"였으므로,
    # 아래 세 제안은 손실 텐서/샘플링만 건드리고 모델 파라미터는 일절 늘리지 않는다.
    # ==================================================================

    # ---- 제안 1. 운동 임계도 가중 FM (Motion-Criticality Weighted FM · AttenA+ 변형) ----
    # GT 행동의 속도 ‖a_h‖ 가 작은(=정밀이 필요한) 구간의 flow 손실을 키운다.
    #   w_h = clip(f(‖a_h‖), 1/clip, clip),  f ∈ {1/v, 1/v², e^{-αv}, 1/log(1+v)}
    # "tgate" 는 우리 변형: flow time t 로 게이트해 저노이즈(t→0, 세부 정밀화) 구간에서만 가중이 살고
    # 고노이즈(t→1, 모드 선택) 구간에선 원래 손실로 복귀한다.  w_h(t) = 1 + (w_h − 1)·(1 − t)
    # off("none")면 곱셈 자체가 실행되지 않아 원본과 비트 단위로 동일.
    action_loss_weighting: str = "none"        # none | inv_speed | inv_speed_tgate
    alw_fn: str = "inv"                        # inv | inv_sq | exp | log
    alw_clip: float = 3.0                      # w 를 [1/clip, clip] 로 제한 (1.0 이면 항등=baseline)
    alw_alpha: float = 5.0                     # alw_fn="exp" 의 감쇠율
    alw_speed_dims: int = 6                    # 속도 계산에 쓸 선행 차원 수 (LIBERO=6, gripper 제외). 0=전체
    alw_normalize: bool = True                 # 배치 평균 w=1 로 정규화 → 유효 학습률 유지
    alw_eps: float = 1e-3                      # ‖a‖=0 방어

    # ---- 제안 2a. 층화 시간 샘플링 (Stratified time sampling) ----
    # Beta(α,1) 의 역CDF 가 u^{1/α} 인 점을 이용해, 배치 내 t 를 iid 대신 층화추출한다.
    # 유효배치 16(원논문 384의 1/24)에서 t 커버리지 편중이 시드 분산의 원인 중 하나라는 가설.
    # beta_beta != 1 이면 역CDF 가 닫힌 형태가 아니므로 자동으로 iid 로 폴백한다.
    time_sampling_stratified: bool = False

    # ---- 제안 2b. 다중 표본 FM (Multi-sample / Antithetic-pair FM) ----
    # prefix(VLM)는 한 번만 돌리고, suffix 에 action 블록을 K개 나란히 붙여 서로 다른 (t, ε) 로
    # 동시에 flow 손실을 매긴다. 블록 간 attention 은 차단하고 position id 를 공유하므로
    # K=1 일 때와 블록 0 의 계산이 정확히 일치한다. VLM 재계산이 없어 비용은 expert 토큰 몇십 개뿐.
    #   fm_antithetic: none(독립 샘플) | time(t₂=1−t₁, ε 공유) | time_noise(t₂=1−t₁, ε₂=−ε₁)
    # ε 를 공유하면 u=ε−a 가 두 샘플에서 동일 → TPC 일관성 항 ‖v¹−v²‖² 이 같은 타깃으로의
    # 분산 감소 정규화가 된다(fm_tpc_weight>0 일 때, Bernoulli(fm_tpc_prob) 게이팅).
    fm_samples: int = 1                        # K. 1 이면 원본 경로 그대로(코드 분기 자체가 안 탐)
    fm_antithetic: str = "none"                # none | time | time_noise
    fm_tpc_weight: float = 0.0                 # 0 이면 일관성 항 없음
    fm_tpc_prob: float = 0.75                  # 일관성 항 적용 확률 (TPC 논문 권장값)

    # ---- 제안 3. Sobolev(시간 미분) 가중 FM 손실 ----
    # 속도 오차 e = v̂ − u 의 chunk 방향 1차 차분에도 벌점을 준다:  ‖e‖² + λ‖Δ_h e‖²
    # FAFM 의 H¹ 정리(1차 미분 정규화 ≡ 주파수 가중 1+λω²)를 표현(DCT) 변경 없이 손실에만 적용.
    # 0 이면 항이 계산되지 않아 원본과 동일.
    action_loss_sobolev_weight: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.lambda_vqa < 0:
            raise ValueError(f"lambda_vqa must be >= 0, got {self.lambda_vqa}")
        if self.action_token_min > self.action_token_max:
            raise ValueError(
                f"action_token_min ({self.action_token_min}) must be <= action_token_max ({self.action_token_max})"
            )
        if self.inference_backend not in {"standard", "optimized"}:
            raise ValueError(
                "inference_backend must be either 'standard' or 'optimized', "
                f"got {self.inference_backend!r}"
            )
        if self.inference_backend == "optimized" and not self.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            use_8bit=self.optimizer_8bit,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        # Memory ω: 과거 action_memory_len개(음수 오프셋)를 추가로 로드. split transform이
        # 초반에 과거/미래로 분리하므로 하류(FAST·모델·정규화)는 미래 청크만 본다.
        if getattr(self, "use_action_memory", False):
            return list(range(-self.action_memory_len, self.chunk_size))
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None:
        n = self.num_video_frames + 1
        return [self.chunk_size * i // (n - 1) for i in range(n)]
