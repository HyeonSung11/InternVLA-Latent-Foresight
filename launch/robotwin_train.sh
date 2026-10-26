#!/usr/bin/env bash
# RoboTwin 1단계 학습 (full_train.sh 파생): 원본 RoboTwin 레시피(60k·유효배치16·chunk50·aloha stats·foresight 토큰 동결)를
# 2×3090 경량화(8-bit AdamW·grad ckpt·micro-batch)로. WAN 조건은 WAN_DEVICE=cuda:1, LF 조건은 LATENT_FORESIGHT=true.
# VRAM 절감: 8-bit AdamW + grad checkpointing + micro-batch. (동결/LoRA 아님)
# 변수: GPU, STEPS, BATCH, GRAD_ACCUM, PORT, TAG, RESIDUAL_PRIOR(true/false), RESIDUAL_PRIOR_SRC(foresight|state)
#  action-loss 연구 플래그(09-15, 전부 기본 off):
#   제안1 ALW=inv_speed|inv_speed_tgate  ALW_FN=inv|inv_sq|exp|log  ALW_CLIP  ALW_DIMS
#   제안2a STRATIFIED=true
#   제안2b FM_SAMPLES=2  FM_ANTI=time|time_noise  FM_TPC=0.1
#   제안3  SOBOLEV=0.1
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
set +u; source ${CONDA_ROOT}/etc/profile.d/conda.sh; conda activate internvla_a1_5; set -u

GPU="${GPU:-0}"
GPUS="${GPUS:-$GPU}"          # 노출할 GPU (예: "0,1" → WAN을 cuda:1로 보낼 수 있음)
WAN_DEVICE="${WAN_DEVICE:-}"  # WAN을 둘 device (예: cuda:1). 빈값이면 본체와 동일.
STEPS="${STEPS:-60000}"       # 원본 RoboTwin 레시피 60k. probe 는 STEPS=60.
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}" # 실제 학습 시 16으로 (유효배치=BATCH×GRAD_ACCUM)
PORT="${PORT:-6600}"
TAG="${TAG:-robotwin_probe}"
FREEZE_LT="${FREEZE_LT:-true}"   # 원본 RoboTwin 레시피는 foresight 토큰 동결(true). LIBERO 런은 false 였음.

export CUDA_VISIBLE_DEVICES=$GPUS
export HF_LEROBOT_HOME=$PWD/data
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DISABLED=true

POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/ckpts/InternVLA-A1.5-base}"
VLM_MODEL_PATH="Qwen/Qwen3.5-2B"
ACTION_TYPE=abs

DATASET_REPO_ID="$(
  find -L "data/robotwin" -mindepth 2 -maxdepth 2 -type d -name "aloha-agilex*" 2>/dev/null \
  | while read -r d; do [[ -d "$d/meta" && -d "$d/videos" ]] && echo "${d#data/}"; done \
  | sort -u | tr '\n' ' ' | xargs)"
[[ -z "${DATASET_REPO_ID}" ]] && { echo "[ERROR] data/robotwin 에 aloha-agilex 데이터셋이 없음"; exit 2; }
echo "datasets: $(echo ${DATASET_REPO_ID} | wc -w)개"

OUTPUT_DIR="outputs/robotwin_train/${TAG}"
echo "=== ROBOTWIN TRAIN ${TAG} | GPU=${GPU} | steps=${STEPS} | batch=${BATCH}×accum${GRAD_ACCUM} | video loss ON | 8-bit ==="

accelerate launch --num_processes=${NUM_PROCESSES:-1} --num_machines=1 --machine_rank=0 \
  --main_process_ip=127.0.0.1 --main_process_port=${PORT} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="${OUTPUT_DIR}" --job_name="${TAG}" --num_workers=4 \
  --policy.type=${POLICY} --policy.repo_id=lerobot_lab/${POLICY} \
  --policy.pretrained_path=${PRETRAINED_PATH} --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=${STEPS} --policy.scheduler_decay_lr=5e-6 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.vlm_model_name_or_path=${VLM_MODEL_PATH} \
  --policy.enable_vqa_loss=true --policy.tokenize_state=true \
  --policy.knowledge_insulation=false \
  --policy.video_loss_only=false --policy.action_loss_only=${ACTION_LOSS_ONLY:-false} \
  --policy.video_loss_weight=1 \
  --policy.freeze_learnable_tokens=${FREEZE_LT} --policy.num_learnable_tokens=50 \
  --policy.freeze_vlm=false --policy.use_vlm_lora=false --policy.use_seam_grounding=${SEAM:-false} \
  --policy.use_latent_foresight=${LATENT_FORESIGHT:-false} \
  --policy.latent_foresight_target=${LF_TARGET:-pixel} --policy.latent_foresight_loss=${LF_LOSS:-mse} \
  --policy.use_action_memory=${ACTION_MEMORY:-false} --policy.action_memory_len=${ACTION_MEMORY_LEN:-8} \
  --policy.use_residual_prior=${RESIDUAL_PRIOR:-false} --policy.residual_prior_source=${RESIDUAL_PRIOR_SRC:-foresight}  --policy.residual_prior_mode=${RESIDUAL_PRIOR_MODE:-velocity} --policy.residual_prior_loss_weight=${RESIDUAL_PRIOR_LW:-1.0} \
  --dataset.use_action_memory=${ACTION_MEMORY:-false} --dataset.action_memory_len=${ACTION_MEMORY_LEN:-8} \
  --policy.action_loss_weighting=${ALW:-none} --policy.alw_fn=${ALW_FN:-inv} --policy.alw_clip=${ALW_CLIP:-3.0} --policy.alw_alpha=${ALW_ALPHA:-5.0} --policy.alw_speed_dims=${ALW_DIMS:-6} --policy.alw_normalize=${ALW_NORM:-true} \
  --policy.time_sampling_stratified=${STRATIFIED:-false} \
  --policy.fm_samples=${FM_SAMPLES:-1} --policy.fm_antithetic=${FM_ANTI:-none} --policy.fm_tpc_weight=${FM_TPC:-0.0} --policy.fm_tpc_prob=${FM_TPC_P:-0.75} \
  --policy.action_loss_sobolev_weight=${SOBOLEV:-0.0} \
  --policy.use_spatial_tokens=${USE_SPATIAL:-false} --policy.spatial_num_tokens=${SP_TOKENS:-64} --policy.spatial_teacher=${SP_TEACHER:-vggt} --policy.spatial_teacher_device=${SP_DEVICE:-} \
  --policy.spatial_loss=${SP_LOSS:-cosine} --policy.spatial_loss_weight=${SP_WEIGHT:-0.15} --policy.spatial_warmup_steps=${SP_WARMUP:-1000} --policy.spatial_isolate=${SP_ISOLATE:-true} \
  --policy.spatial_vggt_layer=${SP_LAYER:--1} --policy.spatial_target_center=${SP_CENTER:-false} \
  --policy.optimizer_8bit=true \
  --policy.wan_device="${WAN_DEVICE}" \
  --dataset.type="${POLICY}" --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.action_mode="${ACTION_TYPE}" --dataset.use_external_stats=true \
  --dataset.external_stats_path=${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/stats.json \
  --dataset.dist_loading=false --dataset.tokenize_state=true \
  --dataset.use_fast_action_tokens=true --dataset.video_backend=pyav \
  --seed=${SEED:-42} --batch_size=${BATCH} --steps=${STEPS} --save_freq=${SAVE_FREQ:-20000} --log_freq=50 \
  --wandb.enable=false
