#!/usr/bin/env bash
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CKPT="${HF_HOME:-$HOME/.cache/huggingface}/ckpts/full_100k_final"
echo "===== EVAL full-100k @ 50 trials (논문 프로토콜) ====="
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} SERVER_ENV=internvla_a1_5 CLIENT_ENV=libero_venv \
LIBERO_HOME=${LIBERO_HOME:-$HOME/LIBERO} VLM_MODEL_PATH=Qwen/Qwen3.5-2B \
STATS_KEY_MODE=suite ROBOT_TYPE_MODE=suite NUM_TRIALS_PER_TASK=50 \
GPU_IDS=0,1 MUJOCO_GL=egl BASE_PORT=5734 CKPT_PATH="${CKPT}" \
EVAL_LOG_DIR="$PWD/outputs/full100k_eval50" \
bash evaluation/LIBERO/run_eval_libero_server_client.sh
echo "ALL_FULL100K_EVAL50_DONE"
