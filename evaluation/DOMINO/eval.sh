#!/usr/bin/env bash
set -euo pipefail

###############################################################################
############################## Evaluation config ###############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOMINO_ROOT="${DOMINO_ROOT:-${REPO_ROOT}/third_party/DOMINO}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/third_party/RoboTwin}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PRETRAINED_CKPT [OUTPUT_PATH] [TASK_CONFIG] [TASK_IDX] [ACTION_TYPE] [HORIZON] [TEST_NUM] [MAX_SETUP_ATTEMPTS] [EXEC_HORIZON] [ACTION_MODE] [DTYPE] [NUM_INFERENCE_STEPS]" >&2
    echo "Example:" >&2
    echo "  $0 /path/to/checkpoint/pretrained_model outputs/domino/run demo_clean_dynamic 8 fm 50 100 100000 10 abs float32 10" >&2
    exit 2
fi

PRETRAINED_CKPT="$1"
OUTPUT_PATH="${2:-${OUTPUT_PATH:-outputs/domino/internvla_a1_5}}"
TASK_CONFIG="${3:-${TASK_CONFIG:-demo_clean_dynamic}}"
TASK_IDX="${4:-${TASK_IDX:-0}}"
ACTION_TYPE="${5:-${ACTION_TYPE:-fm}}"
HORIZON="${6:-${HORIZON:-50}}"
TEST_NUM="${7:-${TEST_NUM:-100}}"
MAX_SETUP_ATTEMPTS="${8:-${MAX_SETUP_ATTEMPTS:-100000}}"
EXEC_HORIZON="${9:-${EXEC_HORIZON:-10}}"
ACTION_MODE="${10:-${ACTION_MODE:-abs}}"
DTYPE="${11:-${DTYPE:-float32}}"
NUM_INFERENCE_STEPS="${12:-${NUM_INFERENCE_STEPS:-0}}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"

OUTPUT_PATH="$("${PYTHON_BIN}" - "${OUTPUT_PATH}" <<'PY_PATH'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY_PATH
)"

if [[ ! -d "${DOMINO_ROOT}" ]]; then
    echo "DOMINO source not found: ${DOMINO_ROOT}" >&2
    echo "Set DOMINO_ROOT or initialize third_party/DOMINO first." >&2
    exit 2
fi

if [[ "${ACTION_MODE}" == "auto" ]]; then
    ACTION_MODE="$("${PYTHON_BIN}" - "${PRETRAINED_CKPT}" <<'PY_AUTO'
import json
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
try:
    cfg = json.loads((ckpt / "train_config.json").read_text())
    print(cfg.get("dataset", {}).get("action_mode") or "abs")
except Exception:
    print("abs")
PY_AUTO
)"
fi

case "${ACTION_MODE}" in
    abs|delta) ;;
    *)
        echo "ERROR: ACTION_MODE must be abs, delta, or auto; got ${ACTION_MODE}" >&2
        exit 2
        ;;
esac
case "${DTYPE}" in
    float32|fp32|bfloat16|bf16) ;;
    *)
        echo "ERROR: DTYPE must be float32/fp32/bfloat16/bf16; got ${DTYPE}" >&2
        exit 2
        ;;
esac
if ! [[ "${NUM_INFERENCE_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_INFERENCE_STEPS must be a non-negative integer; got ${NUM_INFERENCE_STEPS}" >&2
    exit 2
fi

export DOMINO_ROOT ROBOTWIN_ROOT
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${DOMINO_ROOT}:${DOMINO_ROOT}/policy:${DOMINO_ROOT}/description/utils:${DOMINO_ROOT}/script:${ROBOTWIN_ROOT}/envs/curobo/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DOMINO_CACHE_ROOT="${DOMINO_JOB_CACHE_ROOT:-${TMPDIR:-/tmp}/domino_eval_${USER:-user}_$$}"
export TMPDIR="${TMPDIR:-${DOMINO_CACHE_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${DOMINO_CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${DOMINO_CACHE_ROOT}/cuda_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DOMINO_CACHE_ROOT}/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${DOMINO_CACHE_ROOT}/torchinductor_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DOMINO_CACHE_ROOT}/xdg_cache}"
mkdir -p "${OUTPUT_PATH}" "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}"

SAVE_REPLAY_VIDEO="${SAVE_REPLAY_VIDEO:-true}"
SAVE_ENV_VIDEO="${SAVE_ENV_VIDEO:-true}"
extra_args=()
case "${SAVE_REPLAY_VIDEO}" in
    true|1|yes|on) extra_args+=(--args.save-replay-video) ;;
    false|0|no|off) extra_args+=(--args.no-save-replay-video) ;;
    *)
        echo "ERROR: SAVE_REPLAY_VIDEO must be true/false; got ${SAVE_REPLAY_VIDEO}" >&2
        exit 2
        ;;
esac
case "${SAVE_ENV_VIDEO}" in
    true|1|yes|on) extra_args+=(--args.save-env-video) ;;
    false|0|no|off) extra_args+=(--args.no-save-env-video) ;;
    *)
        echo "ERROR: SAVE_ENV_VIDEO must be true/false; got ${SAVE_ENV_VIDEO}" >&2
        exit 2
        ;;
esac
if (( NUM_INFERENCE_STEPS > 0 )); then
    extra_args+=(--args.num_inference_steps "${NUM_INFERENCE_STEPS}")
fi

echo "REPO_ROOT       = ${REPO_ROOT}"
echo "DOMINO_ROOT     = ${DOMINO_ROOT}"
echo "PRETRAINED_CKPT = ${PRETRAINED_CKPT}"
echo "OUTPUT_PATH     = ${OUTPUT_PATH}"
echo "TASK_CONFIG     = ${TASK_CONFIG}"
echo "TASK_IDX        = ${TASK_IDX}"
echo "ACTION_TYPE     = ${ACTION_TYPE}"
echo "ACTION_MODE     = ${ACTION_MODE}"
echo "HORIZON         = ${HORIZON}"
echo "EXEC_HORIZON    = ${EXEC_HORIZON}"
echo "DTYPE           = ${DTYPE}"
echo "NUM_INFERENCE_STEPS = ${NUM_INFERENCE_STEPS}"
echo "INFERENCE_BACKEND = ${INFERENCE_BACKEND}"
echo "TEST_NUM        = ${TEST_NUM}"

cd "${DOMINO_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" \
    --args.ckpt_path "${PRETRAINED_CKPT}" \
    --args.video_dir "${OUTPUT_PATH}" \
    --args.task_config "${TASK_CONFIG}" \
    --args.task_idx "${TASK_IDX}" \
    --args.action_type "${ACTION_TYPE}" \
    --args.action_mode "${ACTION_MODE}" \
    --args.dtype "${DTYPE}" \
    --args.infer_horizon "${HORIZON}" \
    --args.action_horizon_size "${HORIZON}" \
    --args.test_num "${TEST_NUM}" \
    --args.max_setup_attempts "${MAX_SETUP_ATTEMPTS}" \
    --args.execute_horizon "${EXEC_HORIZON}" \
    --args.inference_backend "${INFERENCE_BACKEND}" \
    "${extra_args[@]}"
