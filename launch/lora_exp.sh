#!/usr/bin/env bash
# VLM LoRA 실험 (연구): 동결 병목 완화. base VLM 동결 + 저랭크 adapter만 학습.
# seam_exp.sh 기반. 변수: SEAM(true/false), LORA_R, GPU, STEPS, PORT, TAG
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
set +u; source ${CONDA_ROOT}/etc/profile.d/conda.sh; conda activate internvla_a1_5; set -u

SEAM="${SEAM:-true}"          # seam grounding 병행 여부
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GPU="${GPU:-0}"
STEPS="${STEPS:-30000}"
PORT="${PORT:-6500}"
TAG="${TAG:-lora_seam${SEAM}_r${LORA_R}}"

export CUDA_VISIBLE_DEVICES=$GPU
export HF_LEROBOT_HOME=$PWD/data
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_DISABLED=true

POLICY="internvla_a1_5"
PRETRAINED_PATH="${HF_HOME:-$HOME/.cache/huggingface}/ckpts/InternVLA-A1.5-base"
VLM_MODEL_PATH="Qwen/Qwen3.5-2B"
ACTION_TYPE=abs

DATASET_REPO_ID="$(
  find -L "data/libero" -mindepth 1 -maxdepth 1 -type d -name "libero_*_no_noops*_lerobot" 2>/dev/null \
  | while read -r d; do [[ -d "$d/meta" && -d "$d/videos" ]] && echo "${d#data/}"; done \
  | sort -u | tr '\n' ' ' | xargs)"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
for repo in ${DATASET_REPO_ID}; do
  info_json="data/${repo}/meta/info.json"; subset="$(basename "${repo}")"; subset="${subset%%_no_noops*}"
  python -c "import json;p='${info_json}';d=json.load(open(p));d['robot_type']='${subset}';json.dump(d,open(p,'w'),indent=4)" >/dev/null
done

OUTPUT_DIR="outputs/lora_exp/${TAG}"
echo "=== TRAIN ${TAG} | GPU=${GPU} | steps=${STEPS} | seam=${SEAM} | lora_r=${LORA_R} ==="

accelerate launch --num_processes=1 --num_machines=1 --machine_rank=0 \
  --main_process_ip=127.0.0.1 --main_process_port=${PORT} \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="${OUTPUT_DIR}" --job_name="${TAG}" --num_workers=4 \
  --policy.type=${POLICY} --policy.repo_id=lerobot_lab/${POLICY} \
  --policy.pretrained_path=${PRETRAINED_PATH} --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --policy.chunk_size=8 --policy.n_action_steps=8 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=${STEPS} --policy.scheduler_decay_lr=5e-6 \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.vlm_model_name_or_path=${VLM_MODEL_PATH} \
  --policy.enable_vqa_loss=false --policy.tokenize_state=true \
  --policy.knowledge_insulation=false --policy.video_loss_only=false \
  --policy.action_loss_only=true --policy.freeze_learnable_tokens=true \
  --policy.num_learnable_tokens=50 \
  --policy.freeze_vlm=false \
  --policy.use_vlm_lora=true --policy.lora_r=${LORA_R} --policy.lora_alpha=${LORA_ALPHA} \
  --policy.use_seam_grounding=${SEAM} --policy.seam_num_layers=4 --policy.seam_num_heads=8 \
  --dataset.type="${POLICY}" --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.action_mode="${ACTION_TYPE}" --dataset.use_external_stats=false \
  --dataset.dist_loading=false --dataset.tokenize_state=true \
  --dataset.use_fast_action_tokens=false --dataset.video_backend=pyav \
  --seed=42 --batch_size=8 --steps=${STEPS} --save_freq=5000 --log_freq=50 \
  --wandb.enable=false
