#!/usr/bin/env bash
set -euo pipefail

# Single-GPU smoke-test variant of run_eval_libero_plus.sh. Launches one policy
# server and runs a small task-id range on one suite to validate the full
# server <-> LIBERO-plus path before kicking off a full ~10k-task run.
#
# Usage:
#   GPU_ID=0 TASK_SUITE=libero_goal START_IDX=0 END_IDX=4 \
#     bash evaluation/LIBERO-plus/run_eval_libero_plus_single.sh
#
# See run_eval_libero_plus.sh header for the one-time env + assets setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
SERVER_ENV="${SERVER_ENV:-lerobot_lab}"
CLIENT_ENV="${CLIENT_ENV:-libero_plus}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5784}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
NO_ROTATE_FLAG=""
NO_BINARIZE_FLAG=""
NO_VIDEO_FLAG="${NO_VIDEO_FLAG:-}"
DEBUG_FLAG=""

GPU_ID="${GPU_ID:-0}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-600}"

TASK_SUITE="${TASK_SUITE:-libero_goal}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-4}"

CKPT_PATH="${CKPT_PATH:-}"
LIBERO_HOME="${LIBERO_HOME:-}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
# WAN paths only matter for InternVLA-A1.5 checkpoints when running with
# --no-action_loss_only. The default action_loss_only=True skips WAN loading.
WAN_MODEL_PATH="${WAN_MODEL_PATH:-}"
WAN_VAE_PATH="${WAN_VAE_PATH:-}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
STATS_KEY="${STATS_KEY:-${TASK_SUITE}}"
ROBOT_TYPE="${ROBOT_TYPE:-${TASK_SUITE}}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH is required." >&2
  exit 2
fi
if [[ -z "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME is required (point at the LIBERO-plus repo root)." >&2
  exit 2
fi
TASK_CLASSIFICATION="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
if [[ ! -f "${TASK_CLASSIFICATION}" ]]; then
  echo "[ERROR] task_classification.json not found at ${TASK_CLASSIFICATION}" >&2
  exit 2
fi

EVAL_LOG_DIR="${EVAL_LOG_DIR:-}"
if [[ -z "${EVAL_LOG_DIR}" ]]; then
  EVAL_LOG_DIR="outputs/sim_eval/libero_plus/$(date +%Y%m%d_%H%M%S)_single_gpu${GPU_ID}"
fi
mkdir -p "${EVAL_LOG_DIR}"

# Dedicated LIBERO config dir pointing at the LIBERO-plus repo (NOT ~/.libero,
# which points at stock LIBERO). Without this the package prompts
# interactively on first run and dies with EOFError in the non-interactive
# subshell.
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${EVAL_LOG_DIR}/libero_config}"
mkdir -p "${LIBERO_CONFIG_DIR}"
LP_BENCH_ROOT="${LIBERO_HOME}/libero/libero"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LP_BENCH_ROOT}
bddl_files: ${LP_BENCH_ROOT}/bddl_files
init_states: ${LP_BENCH_ROOT}/init_files
datasets: ${LP_BENCH_ROOT}/../datasets
assets: ${LP_BENCH_ROOT}/assets
YAML

SERVER_LOG="${EVAL_LOG_DIR}/server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "GPU            : ${GPU_ID}"
echo "Port           : ${PORT}"
echo "Suite / shard  : ${TASK_SUITE} [${START_IDX}, ${END_IDX})"
echo "Ckpt           : ${CKPT_PATH}"
echo "LIBERO-plus    : ${LIBERO_HOME}"
echo "Log dir        : ${EVAL_LOG_DIR}"

# ---------------- Start server ----------------
(
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${SERVER_ENV}"
  cd "${PROJ_ROOT}"
  PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" exec python evaluation/LIBERO/policy_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --host "0.0.0.0" \
    --port "${PORT}" \
    --device cuda \
    --resize_size "${RESIZE_SIZE}" \
    --stats_key "${STATS_KEY}" \
    --robot_type "${ROBOT_TYPE}" \
    --vlm_model_path "${VLM_MODEL_PATH}" \
    --wan_model_path "${WAN_MODEL_PATH}" \
    --wan_vae_path "${WAN_VAE_PATH}" \
    ${ACTION_LOSS_ONLY_FLAG} \
    --inference_backend "${INFERENCE_BACKEND}" \
    --idle_timeout -1
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server pid     : ${SERVER_PID}  (log: ${SERVER_LOG})"

# ---------------- Health check ----------------
if ! (
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${SERVER_ENV}"
  cd "${PROJ_ROOT}"
  PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src:${PYTHONPATH:-}" python - <<PY
import time, sys
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy

host = "${HOST}"
port = int("${PORT}")
timeout = int("${SERVER_READY_TIMEOUT_SEC}")
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        client = WebsocketClientPolicy(host=host, port=port)
        resp = client.predict_action({"type": "ping", "request_id": "healthcheck"})
        if resp.get("ok", False):
            print("server ready")
            sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(f"server healthcheck failed after {timeout}s")
PY
) >>"${EVAL_LOG_DIR}/healthcheck.log" 2>&1; then
  echo "[ERROR] server failed to come up (see ${EVAL_LOG_DIR}/healthcheck.log and ${SERVER_LOG})" >&2
  exit 1
fi
echo "Server ready."

# ---------------- Eval one shard ----------------
client_log="${EVAL_LOG_DIR}/client_${TASK_SUITE}_${START_IDX}_${END_IDX}.log"
echo "==== [${TASK_SUITE} ${START_IDX}-${END_IDX}] starting ===="
(
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CLIENT_ENV}"
  cd "${PROJ_ROOT}"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
  # ImageMagick (wand dependency) is in the libero_plus conda env; ensure its
  # shared lib is on the loader path.
  export MAGICK_HOME="${CONDA_PREFIX}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  PYTHONPATH="${LIBERO_HOME}:${PROJ_ROOT}:${PYTHONPATH:-}" \
  python evaluation/LIBERO-plus/eval_libero_plus.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --task_suite_name "${TASK_SUITE}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --num_steps_wait "${NUM_STEPS_WAIT}" \
    --seed "${SEED}" \
    --replan_steps "${REPLAN_STEPS}" \
    --start_idx "${START_IDX}" \
    --end_idx "${END_IDX}" \
    --task_classification_path "${TASK_CLASSIFICATION}" \
    --eval_log_dir "${EVAL_LOG_DIR}" \
    ${NO_ROTATE_FLAG} ${NO_BINARIZE_FLAG} ${NO_VIDEO_FLAG} ${DEBUG_FLAG}
) >"${client_log}" 2>&1 || { echo "[${TASK_SUITE}] FAILED (see ${client_log})" >&2; exit 1; }
echo "==== [${TASK_SUITE} ${START_IDX}-${END_IDX}] done ===="

# ---------------- Aggregate (single shard) ----------------
(
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CLIENT_ENV}"
  cd "${PROJ_ROOT}"
  python evaluation/LIBERO-plus/aggregate_results.py --root "${EVAL_LOG_DIR}"
) || echo "[WARN] aggregation failed; per-shard json under ${EVAL_LOG_DIR}/logs" >&2

echo "Done. Logs: ${EVAL_LOG_DIR}"
