#!/usr/bin/env bash
# 특정 suite 만 50-trial 평가. 사용: bash launch/eval50_suites.sh <TAG> <GPU_IDS> <ROOT> <PORT> <suite...>
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TAG="$1"; GPUS="$2"; ROOT="$3"; PORT="$4"; shift 4; SUITES="$*"
CKPT="$PWD/$ROOT/$TAG/checkpoints/025000/pretrained_model"; OUT="$PWD/outputs/${TAG}_eval50"
echo "[$(date '+%F %T')] EVAL(suites: $SUITES) 시작: $TAG on GPU_IDS=$GPUS"
TASK_SUITES_OVERRIDE="$SUITES" HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} SERVER_ENV=internvla_a1_5 CLIENT_ENV=libero_venv \
LIBERO_HOME=${LIBERO_HOME:-$HOME/LIBERO} VLM_MODEL_PATH=Qwen/Qwen3.5-2B \
STATS_KEY_MODE=suite ROBOT_TYPE_MODE=suite NUM_TRIALS_PER_TASK=50 \
GPU_IDS="$GPUS" MUJOCO_GL=egl BASE_PORT=${PORT} CKPT_PATH="${CKPT}" EVAL_LOG_DIR="${OUT}" \
bash evaluation/LIBERO/run_eval_libero_server_client_suites.sh
echo "[$(date '+%F %T')] EVAL(suites) 종료: $TAG rc=$?"
