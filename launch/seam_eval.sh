#!/usr/bin/env bash
# Seam Grounding ablation: LIBERO eval of seam_on vs seam_off (15k ckpt).
# 재현과 동일 셋업(2-GPU, suite stats), CKPT_PATH만 교체해 순차 실행.
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
STEP="${STEP:-015000}"
TRIALS="${TRIALS:-20}"

for v in seam_on seam_off; do
  CKPT="$PWD/outputs/seam_exp/${v}/checkpoints/${STEP}/pretrained_model"
  echo "=========================================================="
  echo "===== EVAL ${v}  (ckpt step=${STEP}, trials=${TRIALS}) ====="
  echo "CKPT=${CKPT}"
  echo "=========================================================="
  HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
  SERVER_ENV=internvla_a1_5 CLIENT_ENV=libero_venv \
  LIBERO_HOME=${LIBERO_HOME:-$HOME/LIBERO} \
  VLM_MODEL_PATH=Qwen/Qwen3.5-2B \
  STATS_KEY_MODE=suite ROBOT_TYPE_MODE=suite \
  NUM_TRIALS_PER_TASK="${TRIALS}" \
  GPU_IDS=0,1 MUJOCO_GL=egl \
  BASE_PORT=5734 \
  CKPT_PATH="${CKPT}" \
  EVAL_LOG_DIR="$PWD/outputs/seam_eval/${v}" \
  bash evaluation/LIBERO/run_eval_libero_server_client.sh
  echo "===== ${v} eval exit=$? ====="
done
echo "ALL_EVAL_DONE"
