#!/usr/bin/env bash
# 체크포인트 하나를 LIBERO 50-trial 프로토콜로 평가 (seed_eval_queue.sh 와 동일 설정).
# 사용: bash launch/eval50_one.sh <TAG> [ROOT=outputs/full_train] [PORT]
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TAG="$1"; ROOT="${2:-outputs/full_train}"; PORT="${3:-5800}"
# 25k 가 아닌 런은 CKPT_STEP 으로 체크포인트 step 을 지정한다 (예: CKPT_STEP=100000)
CKPT="$PWD/$ROOT/$TAG/checkpoints/${CKPT_STEP:-025000}/pretrained_model"; OUT="$PWD/outputs/${TAG}_eval50"
[[ -d "$CKPT" ]] || { echo "[ERROR] 없음: $CKPT"; exit 2; }
[[ -f "$OUT/libero_10/results_libero_10.json" ]] && { echo "[SKIP] $TAG 이미 평가됨"; exit 0; }
echo "[$(date '+%F %T')] EVAL 시작: $TAG"
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} SERVER_ENV=internvla_a1_5 CLIENT_ENV=libero_venv \
LIBERO_HOME=${LIBERO_HOME:-$HOME/LIBERO} VLM_MODEL_PATH=Qwen/Qwen3.5-2B \
STATS_KEY_MODE=suite ROBOT_TYPE_MODE=suite NUM_TRIALS_PER_TASK=50 \
GPU_IDS="${GPU_IDS:-0,1}" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_IDS%%,*}}" BASE_PORT=${PORT} CKPT_PATH="${CKPT}" EVAL_LOG_DIR="${OUT}" \
bash evaluation/LIBERO/run_eval_libero_server_client.sh
echo "[$(date '+%F %T')] EVAL 종료: $TAG rc=$?"
