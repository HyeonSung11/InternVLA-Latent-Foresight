#!/usr/bin/env bash
# LoRA 체크포인트 LIBERO eval. STEPS_LIST에 나열된 스텝을 순차 평가.
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
RUN="${RUN:-seam_lora_r16_50k}"
STEPS_LIST="${STEPS_LIST:-050000 015000}"
TRIALS="${TRIALS:-20}"
for step in ${STEPS_LIST}; do
  CKPT="$PWD/outputs/lora_exp/${RUN}/checkpoints/${step}/pretrained_model"
  [ -d "$CKPT" ] || { echo "SKIP ${step}: no ckpt"; continue; }
  echo "===== EVAL ${RUN} @ step ${step} (trials=${TRIALS}) ====="
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} SERVER_ENV=internvla_a1_5 CLIENT_ENV=libero_venv \
  LIBERO_HOME=${LIBERO_HOME:-$HOME/LIBERO} VLM_MODEL_PATH=Qwen/Qwen3.5-2B \
  STATS_KEY_MODE=suite ROBOT_TYPE_MODE=suite NUM_TRIALS_PER_TASK="${TRIALS}" \
  GPU_IDS=0,1 MUJOCO_GL=egl BASE_PORT=5834 CKPT_PATH="${CKPT}" \
  EVAL_LOG_DIR="$PWD/outputs/lora_eval/${RUN}_step${step}" \
  bash evaluation/LIBERO/run_eval_libero_server_client.sh
  echo "===== ${step} exit=$? ====="
done
echo "ALL_LORA_EVAL_DONE"
