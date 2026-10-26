#!/usr/bin/env bash
# Seam Grounding ablation finetune on LIBERO (연구 실험).
# 원본 finetune_libero.sh 기반. VLM/foresight/WAN 전부 동결 → Expert(+seam)만 학습.
# 변수: SEAM(true/false), GPU(0/1), STEPS, PORT
set -uo pipefail

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
set +u; source ${CONDA_ROOT}/etc/profile.d/conda.sh; conda activate internvla_a1_5; set -u

SEAM="${SEAM:-true}"
GPU="${GPU:-0}"
STEPS="${STEPS:-15000}"
PORT="${PORT:-6400}"

export CUDA_VISIBLE_DEVICES=$GPU
export HF_LEROBOT_HOME=$PWD/data          # data/libero/<repo> 를 repo_id로 찾게 함
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 단편화 완화
export WANDB_MODE=offline WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0

POLICY="internvla_a1_5"
PRETRAINED_PATH="${HF_HOME:-$HOME/.cache/huggingface}/ckpts/InternVLA-A1.5-base"
VLM_MODEL_PATH="Qwen/Qwen3.5-2B"
ACTION_TYPE=abs

# ---- 데이터셋 repo_id 목록 (data/ 접두 제거) ----
DATASET_REPO_ID="$(
  find -L "data/libero" -mindepth 1 -maxdepth 1 -type d -name "libero_*_no_noops*_lerobot" 2>/dev/null \
  | while read -r d; do [[ -d "$d/meta" && -d "$d/videos" ]] && echo "${d#data/}"; done \
  | sort -u | tr '\n' ' ' | xargs
)"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"

# ---- info.json robot_type 를 suite별로 패치 (stats 충돌 방지) ----
for repo in ${DATASET_REPO_ID}; do
  info_json="data/${repo}/meta/info.json"
  subset="$(basename "${repo}")"; subset="${subset%%_no_noops*}"
  python -c "
import json; p='${info_json}'
d=json.load(open(p))
if d.get('robot_type')!='${subset}':
    d['robot_type']='${subset}'; json.dump(d,open(p,'w'),indent=4); print('[patch]', '${subset}')
else: print('[patch] already', '${subset}')"
done

TAG=$([ "$SEAM" = "true" ] && echo "seam_on" || echo "seam_off")
OUTPUT_DIR="outputs/seam_exp/${TAG}"
echo "=== TRAIN ${TAG} | GPU=${GPU} | steps=${STEPS} | out=${OUTPUT_DIR} ==="

accelerate launch --num_processes=1 --num_machines=1 --machine_rank=0 \
  --main_process_ip=127.0.0.1 --main_process_port=${PORT} \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${TAG}" \
  --num_workers=4 \
  --policy.type=${POLICY} \
  --policy.repo_id=lerobot_lab/${POLICY} \
  --policy.pretrained_path=${PRETRAINED_PATH} \
  --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=${STEPS} \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.vlm_model_name_or_path=${VLM_MODEL_PATH} \
  --policy.enable_vqa_loss=false \
  --policy.tokenize_state=true \
  --policy.knowledge_insulation=false \
  --policy.video_loss_only=false \
  --policy.action_loss_only=true \
  --policy.freeze_learnable_tokens=true \
  --policy.num_learnable_tokens=50 \
  --policy.freeze_vlm=true \
  --policy.use_seam_grounding=${SEAM} \
  --policy.seam_num_layers=4 \
  --policy.seam_num_heads=8 \
  --dataset.type="${POLICY}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.action_mode="${ACTION_TYPE}" \
  --dataset.use_external_stats=false \
  --dataset.dist_loading=false \
  --dataset.tokenize_state=true \
  --dataset.use_fast_action_tokens=false \
  --dataset.video_backend=pyav \
  --seed=42 \
  --batch_size=8 \
  --steps=${STEPS} \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=false
