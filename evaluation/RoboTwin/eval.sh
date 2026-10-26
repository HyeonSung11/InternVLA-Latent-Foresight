#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################

CONDA_ROOT=${_CONDA_ROOT}
CONDA_ENV=internvla_a1_5

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export HF_HOME=${HF_HOME}

###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
PRETRAINED_CKPT="${1:-${PRETRAINED_CKPT:-InternRobotics/InternVLA-A1.5-RoboTwin}}"
TASK_CONFIG="${3:-${TASK_CONFIG:-demo_clean}}"
TASK_IDX="${4:-${TASK_IDX:-44}}"
OUTPUT_PATH="${2:-${OUTPUT_PATH:-outputs/robotwin/internvla_a1_5/${TASK_CONFIG}/${TASK_IDX}}}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
ACTION_MODE="${ACTION_MODE:-abs}"
INFER_HORIZON="${INFER_HORIZON:-20}"

if [[ -z "${PRETRAINED_CKPT}" ]]; then
  echo "Usage: bash evaluation/RoboTwin/eval.sh <checkpoint> [output_path] [task_config] [task_idx]" >&2
  exit 2
fi

mkdir -p "${OUTPUT_PATH}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd ${REPO_ROOT}/third_party/RoboTwin

python ../../evaluation/RoboTwin/inference.py \
  --ckpt-path "${PRETRAINED_CKPT}" \
  --video-dir "${OUTPUT_PATH}" \
  --task-config "${TASK_CONFIG}" \
  --task-idx "${TASK_IDX}" \
  --action-mode "${ACTION_MODE}" \
  --infer-horizon "${INFER_HORIZON}" \
  --inference-backend "${INFERENCE_BACKEND}" \
