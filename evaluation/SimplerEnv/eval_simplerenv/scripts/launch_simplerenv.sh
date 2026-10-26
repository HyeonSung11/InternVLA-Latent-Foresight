#!/usr/bin/env bash
# ============================================================================
# Launch N parallel SimplerEnv evaluation clients, each connecting to the
# matching policy server (port = BASE_PORT + i).
#
# Usage:
#   bash evaluation/SimplerEnv/eval_simplerenv/scripts/launch_simplerenv.sh \
#       <NUM_PORTS> <TASK> <LOG_PATH>
#
# Example:
#   bash evaluation/SimplerEnv/eval_simplerenv/scripts/launch_simplerenv.sh \
#       8 bridge ./logs
#
# Optional env vars:
#   CONDA_ROOT       : conda installation root (default: $HOME/miniconda3)
#   SIM_ENV          : conda env with SimplerEnv (default: simplerenv_state)
#   SimplerEnv_PATH  : SimplerEnv repo path (used to resolve rgb_overlay_path
#                      strings inside env_options/*.json). REQUIRED.
#   BASE_PORT        : first port to dial (default: 10086)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJ_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
SIM_ENV="${SIM_ENV:-simplerenv_state}"
SimplerEnv_PATH="${SimplerEnv_PATH:-}"
BASE_PORT="${BASE_PORT:-10086}"

NUM_PORTS="${1:-8}"
TASK="${2:-bridge}"
LOG_PATH="${3:-./logs}"

if [ -z "${NUM_PORTS}" ] || [ -z "${TASK}" ] || [ -z "${LOG_PATH}" ]; then
    echo "Usage: $0 <NUM_PORTS> <TASK> <LOG_PATH>"
    exit 1
fi
if [ -z "${SimplerEnv_PATH}" ]; then
    echo "Error: SimplerEnv_PATH must point at the SimplerEnv (allenzren) checkout."
    exit 1
fi
if [ ! -d "${SimplerEnv_PATH}" ]; then
    echo "Error: SimplerEnv_PATH does not exist: ${SimplerEnv_PATH}"
    exit 1
fi

mkdir -p "${LOG_PATH}/${TASK}"

echo "============================================"
echo "SimplerEnv evaluation clients"
echo "  Project root  : ${PROJ_ROOT}"
echo "  Eval dir      : ${EVAL_DIR}"
echo "  SimplerEnv    : ${SimplerEnv_PATH}"
echo "  Conda env     : ${SIM_ENV}"
echo "  Task          : ${TASK}"
echo "  Workers       : ${NUM_PORTS}"
echo "  Base port     : ${BASE_PORT}"
echo "  Log path      : ${LOG_PATH}/${TASK}"
echo "============================================"

declare -a job_pids=()
for i in $(seq 0 $((NUM_PORTS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_PATH}/${TASK}/${i}_eval.log"

    echo "Starting worker ${i} on port ${PORT} -> ${LOG_FILE}"

    # DISPLAY="" + Vulkan ICD env so SAPIEN renders headlessly.
    DISPLAY="" \
    VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
    CUDA_VISIBLE_DEVICES="${i}" \
    bash -c "
        source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${SIM_ENV} && \
        PYTHONPATH=${PROJ_ROOT}:${PROJ_ROOT}/src:${SimplerEnv_PATH}:\${PYTHONPATH:-} \
        python -u ${EVAL_DIR}/main.py \
            --args.dataset-name ${TASK} \
            --args.experiment-root ${LOG_PATH}/${TASK} \
            --args.worker-id ${i} \
            --args.num-workers ${NUM_PORTS} \
            --args.port ${PORT} \
            --args.simpler-env-root ${SimplerEnv_PATH}
    " 2>&1 | tee "${LOG_FILE}" &
    job_pids+=("$!")
    echo "  worker ${i} pid=$!"
done

echo "Waiting for ${#job_pids[@]} workers to finish..."
wait
echo "All workers finished. Logs and CSVs under: ${LOG_PATH}/${TASK}"
