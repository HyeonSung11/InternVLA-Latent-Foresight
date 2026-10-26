#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------- User-configurable defaults ----------------
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
SERVER_ENV="${SERVER_ENV:-lerobot_lab}"
CLIENT_ENV="${CLIENT_ENV:-libero_venv}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-5734}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
NO_ROTATE_FLAG=""
NO_BINARIZE_FLAG=""
NO_VIDEO_FLAG=""
DEBUG_FLAG=""

# GPU scheduling: detect free GPUs by free memory, run one server per GPU, and
# dispatch task suites from a shared queue (one task per GPU at a time).
GPU_MEM_FREE_THRESHOLD_MB="${GPU_MEM_FREE_THRESHOLD_MB:-30000}"
# Optional explicit override, comma-separated (e.g. GPU_IDS=0,2,5). Skips auto-detection.
GPU_IDS="${GPU_IDS:-}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-600}"

# All four LIBERO task suites are evaluated in parallel across free GPUs.
TASK_SUITES=(${TASK_SUITES_OVERRIDE:-libero_spatial libero_object libero_goal libero_10})

# Required paths — override via env vars.
CKPT_PATH="${CKPT_PATH:-}"
LIBERO_HOME="${LIBERO_HOME:-}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
# WAN paths only matter for InternVLA-A1.5 checkpoints when running with
# --no-action_loss_only. The default action_loss_only=True skips WAN loading.
WAN_MODEL_PATH="${WAN_MODEL_PATH:-}"
WAN_VAE_PATH="${WAN_VAE_PATH:-}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
# Per-suite stats_key / robot_type: when set to `suite`, uses the task suite
# name for both. Set to a fixed string (e.g. panda) to reuse one key across
# suites — matches the public lerobot/libero dataset stats.json.
STATS_KEY_MODE="${STATS_KEY_MODE:-panda}"
ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-panda}"

EVAL_LOG_DIR="${EVAL_LOG_DIR:-}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH is required. Set CKPT_PATH=/path/to/ckpt or edit this script." >&2
  exit 2
fi
if [[ -z "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME is required. Set LIBERO_HOME=/path/to/LIBERO or edit this script." >&2
  exit 2
fi

if [[ -z "${EVAL_LOG_DIR}" ]]; then
  EVAL_LOG_DIR="outputs/sim_eval/libero/$(date +%Y%m%d_%H%M%S)_all"
fi
mkdir -p "${EVAL_LOG_DIR}"

QUEUE_FILE="${EVAL_LOG_DIR}/task_queue.txt"
QUEUE_LOCK="${EVAL_LOG_DIR}/queue.lock"
PIDS_FILE="${EVAL_LOG_DIR}/server_pids.txt"
WORKER_PIDS_FILE="${EVAL_LOG_DIR}/worker_pids.txt"
: > "${PIDS_FILE}"
: > "${WORKER_PIDS_FILE}"
printf '%s\n' "${TASK_SUITES[@]}" > "${QUEUE_FILE}"

cleanup() {
  # Kill workers first so they stop dispatching new client runs, then servers.
  for f in "${WORKER_PIDS_FILE}" "${PIDS_FILE}"; do
    [[ -f "${f}" ]] || continue
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      kill "${pid}" >/dev/null 2>&1 || true
    done < "${f}"
  done
}
trap cleanup EXIT INT TERM

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# ---------------- GPU discovery ----------------
detect_free_gpus() {
  if [[ -n "${GPU_IDS}" ]]; then
    echo "${GPU_IDS}" | tr ',' '\n' | sed '/^$/d'
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -v t="${GPU_MEM_FREE_THRESHOLD_MB}" -F', *' '$2+0 >= t {print $1}'
}

mapfile -t FREE_GPUS < <(detect_free_gpus)
NUM_TASKS=${#TASK_SUITES[@]}
NUM_FREE=${#FREE_GPUS[@]}

if (( NUM_FREE == 0 )); then
  echo "[ERROR] No free GPUs detected (threshold=${GPU_MEM_FREE_THRESHOLD_MB}MB). Set GPU_IDS=... to override." >&2
  exit 1
fi

NUM_WORKERS=${NUM_FREE}
if (( NUM_WORKERS > NUM_TASKS )); then
  NUM_WORKERS=${NUM_TASKS}
fi
WORKER_GPUS=("${FREE_GPUS[@]:0:${NUM_WORKERS}}")

echo "Free GPUs detected: ${FREE_GPUS[*]}"
echo "Using ${NUM_WORKERS} worker(s) on GPU(s): ${WORKER_GPUS[*]}"
echo "Task suites (${NUM_TASKS}): ${TASK_SUITES[*]}"
echo "Log dir: ${EVAL_LOG_DIR}"

resolve_stats_key() {
  local task="$1"
  if [[ "${STATS_KEY_MODE}" == "suite" ]]; then
    echo "${task}"
  else
    echo "${STATS_KEY_MODE}"
  fi
}

resolve_robot_type() {
  local task="$1"
  if [[ "${ROBOT_TYPE_MODE}" == "suite" ]]; then
    echo "${task}"
  else
    echo "${ROBOT_TYPE_MODE}"
  fi
}

# ---------------- Worker definition ----------------
# A worker pops tasks from QUEUE_FILE (flock-protected). For each task it
# spins up a dedicated server bound to that task's stats/robot_type, runs the
# eval client, then tears the server down before grabbing the next task.
run_worker() {
  local gpu_id="$1"
  local worker_idx="$2"
  local port=$((BASE_PORT + worker_idx))

  local worker_dir="${EVAL_LOG_DIR}/worker_gpu${gpu_id}"
  mkdir -p "${worker_dir}"
  local worker_log="${worker_dir}/worker.log"

  while true; do
    local task=""
    {
      flock 9
      local line
      line=$(head -n 1 "${QUEUE_FILE}" 2>/dev/null || true)
      if [[ -n "${line}" ]]; then
        sed -i '1d' "${QUEUE_FILE}"
      fi
      task="${line}"
    } 9>"${QUEUE_LOCK}"
    if [[ -z "${task}" ]]; then
      break
    fi

    local task_dir="${EVAL_LOG_DIR}/${task}"
    mkdir -p "${task_dir}"
    local server_log="${task_dir}/server.log"
    local client_log="${task_dir}/libero_eval.log"

    local stats_key
    stats_key=$(resolve_stats_key "${task}")
    local robot_type
    robot_type=$(resolve_robot_type "${task}")

    echo "[worker gpu=${gpu_id} port=${port}] starting server for ${task} (stats_key=${stats_key})" | tee -a "${worker_log}"

    (
      source "${CONDA_ROOT}/etc/profile.d/conda.sh"
      set +eu
      conda activate "${SERVER_ENV}"
      cd "${PROJ_ROOT}"
      PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src:${PYTHONPATH:-}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" exec python evaluation/LIBERO/policy_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --host "0.0.0.0" \
        --port "${port}" \
        --device cuda \
        --resize_size "${RESIZE_SIZE}" \
        --stats_key "${stats_key}" \
        --robot_type "${robot_type}" \
        --vlm_model_path "${VLM_MODEL_PATH}" \
        --wan_model_path "${WAN_MODEL_PATH}" \
        --wan_vae_path "${WAN_VAE_PATH}" \
        ${ACTION_LOSS_ONLY_FLAG} \
        --inference_backend "${INFERENCE_BACKEND}" \
        --idle_timeout -1
    ) >"${server_log}" 2>&1 &
    local server_pid=$!
    echo "${server_pid}" >> "${PIDS_FILE}"

    # Health-check: wait until the server logs that it is listening (robust on this
    # setup; the previous websocket ping loop could wedge intermittently).
    if ! (
      deadline=$(( $(date +%s) + SERVER_READY_TIMEOUT_SEC ))
      while [ "$(date +%s)" -lt "${deadline}" ]; do
        if grep -q "server listening on 0.0.0.0:${port}" "${server_log}" 2>/dev/null; then
          echo "server ready (detected 'listening' in server log)"
          sleep 3
          exit 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
          echo "server process died during startup"
          exit 1
        fi
        sleep 2
      done
      echo "server did not report listening within ${SERVER_READY_TIMEOUT_SEC}s"
      exit 1
    ) >>"${worker_log}" 2>&1; then
      echo "[worker gpu=${gpu_id}] [${task}] server failed to come up; skipping task" | tee -a "${worker_log}" >&2
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" 2>/dev/null || true
      continue
    fi

    echo "[worker gpu=${gpu_id} port=${port}] server ready (pid=${server_pid}) for ${task}" | tee -a "${worker_log}"

    echo "==== [gpu=${gpu_id}] [${task}] starting ====" | tee -a "${worker_log}"
    (
      source "${CONDA_ROOT}/etc/profile.d/conda.sh"
      set +eu
      conda activate "${CLIENT_ENV}"
      cd "${PROJ_ROOT}"
      export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
      PYTHONPATH="${PROJ_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}" \
      python evaluation/LIBERO/eval_libero_server_client.py \
        --host "127.0.0.1" \
        --port "${port}" \
        --task_suite_name "${task}" \
        --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
        --max_tasks "${MAX_TASKS}" \
        --num_steps_wait "${NUM_STEPS_WAIT}" \
        --seed "${SEED}" \
        --replan_steps "${REPLAN_STEPS}" \
        --eval_log_dir "${task_dir}" \
        ${NO_ROTATE_FLAG} ${NO_BINARIZE_FLAG} ${NO_VIDEO_FLAG} ${DEBUG_FLAG}
    ) >"${client_log}" 2>&1 || echo "[worker gpu=${gpu_id}] [${task}] FAILED (see ${client_log})" | tee -a "${worker_log}" >&2
    echo "==== [gpu=${gpu_id}] [${task}] done ====" | tee -a "${worker_log}"

    echo "[worker gpu=${gpu_id}] [${task}] stopping server (pid=${server_pid})" | tee -a "${worker_log}"
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" 2>/dev/null || true
  done

  echo "[worker gpu=${gpu_id}] queue drained" | tee -a "${worker_log}"
}

# ---------------- Spawn workers ----------------
WORKER_PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
  run_worker "${WORKER_GPUS[$i]}" "$i" &
  WORKER_PIDS+=($!)
  echo "$!" >> "${WORKER_PIDS_FILE}"
done

EXIT_CODE=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    EXIT_CODE=1
  fi
done

echo "Done. Logs: ${EVAL_LOG_DIR}"
exit "${EXIT_CODE}"
