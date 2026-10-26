#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOMINO_ROOT="${DOMINO_ROOT:-${REPO_ROOT}/third_party/DOMINO}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/third_party/RoboTwin}"
SINGLE_EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
if [[ ${NUM_GPUS} -lt 1 ]]; then
    echo "ERROR: GPUS cannot be empty" >&2
    exit 2
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 PRETRAINED_CKPT OUTPUT_ROOT [TASK_CONFIG] [TASK_START] [TASK_END] [TEST_NUM] [MAX_SETUP_ATTEMPTS] [EXEC_HORIZON] [ACTION_MODE] [DTYPE] [NUM_INFERENCE_STEPS] [HORIZON]" >&2
    echo "Example:" >&2
    echo "  GPUS=0,1,2,3,4,5,6,7 $0 /path/to/checkpoint/pretrained_model outputs/domino/run demo_clean_dynamic 0 34 100 100000 10 abs float32 10 50" >&2
    exit 2
fi

PRETRAINED_CKPT="$1"
OUTPUT_ROOT="$2"
TASK_CONFIG="${3:-${TASK_CONFIG:-demo_clean_dynamic}}"
TASK_START="${4:-0}"
TASK_END="${5:-34}"
TEST_NUM="${6:-${TEST_NUM:-100}}"
MAX_SETUP_ATTEMPTS="${7:-${MAX_SETUP_ATTEMPTS:-100000}}"
EXEC_HORIZON="${8:-${EXEC_HORIZON:-10}}"
ACTION_MODE="${9:-${ACTION_MODE:-abs}}"
DTYPE="${10:-${DTYPE:-float32}}"
NUM_INFERENCE_STEPS="${11:-${NUM_INFERENCE_STEPS:-0}}"
HORIZON="${12:-${HORIZON:-50}}"
ACTION_TYPE="${ACTION_TYPE:-fm}"

OUTPUT_ROOT="$("${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY_PATH'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve())
PY_PATH
)"

if (( TASK_START < 0 || TASK_END < TASK_START || TASK_END > 34 )); then
    echo "ERROR: invalid task range ${TASK_START}..${TASK_END}; valid range is 0..34" >&2
    exit 2
fi

task_names=(
    "adjust_bottle"
    "beat_block_hammer"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_can_basket"
    "place_container_plate"
    "place_empty_cup"
    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle"
    "shake_bottle_horizontally"
    "stamp_seal"
)

LOG_ROOT="${OUTPUT_ROOT}/logs/${TASK_CONFIG}"
mkdir -p "${LOG_ROOT}"

JOB_CACHE_ROOT="${DOMINO_JOB_CACHE_ROOT:-${TMPDIR:-/tmp}/domino_eval_${USER:-user}_${TASK_START}_${TASK_END}_$$}"
SCHED_ROOT="${JOB_CACHE_ROOT}/scheduler"
NEXT_TASK_FILE="${SCHED_ROOT}/next_task"
LOCK_DIR="${SCHED_ROOT}/lock"
FAIL_FILE="${SCHED_ROOT}/failed"
mkdir -p "${SCHED_ROOT}"
printf '%s\n' "${TASK_START}" >"${NEXT_TASK_FILE}"
export DOMINO_JOB_CACHE_ROOT="${JOB_CACHE_ROOT}"

echo "PRETRAINED_CKPT = ${PRETRAINED_CKPT}"
echo "OUTPUT_ROOT     = ${OUTPUT_ROOT}"
echo "TASK_CONFIG     = ${TASK_CONFIG}"
echo "TASK_RANGE      = ${TASK_START}..${TASK_END}"
echo "GPUS            = ${GPUS}"
echo "HORIZON         = ${HORIZON}"
echo "EXEC_HORIZON    = ${EXEC_HORIZON}"
echo "ACTION_MODE     = ${ACTION_MODE}"
echo "DTYPE           = ${DTYPE}"
echo "NUM_INFERENCE_STEPS = ${NUM_INFERENCE_STEPS}"
echo "LOG_ROOT        = ${LOG_ROOT}"

PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" >/dev/null 2>&1 || true
    done
    rm -rf "${SCHED_ROOT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

claim_task() {
    local task_idx
    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
        sleep 0.2
    done
    task_idx="$(cat "${NEXT_TASK_FILE}" 2>/dev/null || true)"
    if [[ ! "${task_idx}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid scheduler state: ${task_idx}" >&2
        rmdir "${LOCK_DIR}" 2>/dev/null || true
        return 1
    fi
    if (( task_idx > TASK_END )); then
        rmdir "${LOCK_DIR}"
        return 1
    fi
    printf '%s\n' "$((task_idx + 1))" >"${NEXT_TASK_FILE}"
    rmdir "${LOCK_DIR}"
    printf '%s\n' "${task_idx}"
}

worker_loop() {
    local worker_idx="$1"
    local gpu="$2"
    local task_idx
    local task_name
    local output_path
    local log_file

    while task_idx="$(claim_task)"; do
        task_name="${task_names[$task_idx]}"
        output_path="${OUTPUT_ROOT}/${task_name}/${TASK_CONFIG}"
        log_file="${LOG_ROOT}/${task_name}.log"

        echo "worker=${worker_idx} gpu=${gpu} task=${task_idx}:${task_name} log=${log_file}"
        if ! (
            CUDA_VISIBLE_DEVICES="${gpu}" \
            DOMINO_ROOT="${DOMINO_ROOT}" \
            ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" \
            bash "${SINGLE_EVAL_SCRIPT}" \
                "${PRETRAINED_CKPT}" \
                "${output_path}" \
                "${TASK_CONFIG}" \
                "${task_idx}" \
                "${ACTION_TYPE}" \
                "${HORIZON}" \
                "${TEST_NUM}" \
                "${MAX_SETUP_ATTEMPTS}" \
                "${EXEC_HORIZON}" \
                "${ACTION_MODE}" \
                "${DTYPE}" \
                "${NUM_INFERENCE_STEPS}"
        ) >"${log_file}" 2>&1; then
            {
                echo "worker=${worker_idx} gpu=${gpu} task_idx=${task_idx} task_name=${task_name}"
                echo "log=${log_file}"
            } >>"${FAIL_FILE}"
        fi
    done
}

WARMUP_CUROBO="${WARMUP_CUROBO:-true}"
case "${WARMUP_CUROBO}" in
    true|1|yes|on)
        WARMUP_LOG="${LOG_ROOT}/_curobo_warmup.log"
        echo "Warming up curobo on GPU ${GPU_ARR[0]}"
        if ! (
            export CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}"
            export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${DOMINO_ROOT}:${DOMINO_ROOT}/policy:${DOMINO_ROOT}/description/utils:${DOMINO_ROOT}/script:${ROBOTWIN_ROOT}/envs/curobo/src:${PYTHONPATH:-}"
            cd "${DOMINO_ROOT}"
            python - <<'PY_WARMUP'
import importlib
print("curobo warmup: importing envs.robot.planner", flush=True)
importlib.import_module("envs.robot.planner")
print("curobo warmup: done", flush=True)
PY_WARMUP
        ) >"${WARMUP_LOG}" 2>&1; then
            echo "Curobo warmup failed; log=${WARMUP_LOG}" >&2
            tail -n 80 "${WARMUP_LOG}" >&2 || true
            exit 1
        fi
        ;;
    false|0|no|off)
        echo "Skipping curobo warmup"
        ;;
    *)
        echo "ERROR: WARMUP_CUROBO must be true/false; got ${WARMUP_CUROBO}" >&2
        exit 2
        ;;
esac

for worker_idx in "${!GPU_ARR[@]}"; do
    worker_loop "${worker_idx}" "${GPU_ARR[$worker_idx]}" &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

if [[ -f "${FAIL_FILE}" ]]; then
    echo "Some DOMINO tasks failed:" >&2
    cat "${FAIL_FILE}" >&2
    exit 1
fi

echo "All DOMINO tasks finished. Logs: ${LOG_ROOT}"
