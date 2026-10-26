#!/usr/bin/env bash
# ============================================================================
# Launch N parallel policy servers
# (`evaluation/LIBERO/policy_server/server_policy.py`), one per (GPU, port).
# Pairs with `launch_simplerenv.sh` which expects each rank `i` to dial
# `BASE_PORT + i`.
#
# Supports both PI05 and InternVLA-A1.5 checkpoints; the server auto-
# dispatches on the saved config type. For InternVLA-A1.5 checkpoints running
# with `--no-action_loss_only`, pass `WAN_MODEL_PATH` / `WAN_VAE_PATH` so the
# policy can load the WAN DiT + VAE weights at init time. The default
# `--action_loss_only` skips WAN loading, which matches the public InternVLA-
# A1.5 release.
#
# Bridge dataset uses `robot_type=widowx`; we default --stats_key widowx /
# --robot_type widowx so `base_backend._pick_stats_key` selects the bridge
# subset deterministically when the checkpoint stats.json holds multiple keys.
#
# Usage:
#   bash evaluation/SimplerEnv/eval_simplerenv/scripts/deploy.sh \
#       <CKPT_PATH> <NUM_PORTS> <NUM_GPUS>
#
# Example (InternVLA-A1.5 bridge checkpoint):
#   VLM_MODEL_PATH=/path/to/Qwen3.5-2B-Action \
#   bash evaluation/SimplerEnv/eval_simplerenv/scripts/deploy.sh \
#       /path/to/pretrained_model 8 8
#
# Optional env vars:
#   CONDA_ROOT            : default $HOME/miniconda3
#   SERVER_ENV            : default lerobot_lab (env that runs the policy server)
#   BASE_PORT             : default 10086 (must match launch_simplerenv.sh)
#   SESSION_NAME          : default policy_servers (tmux session)
#   RESIZE_SIZE           : default 224
#   VLM_MODEL_PATH        : override for the chat-processor backbone.
#   WAN_MODEL_PATH        : override for the WAN checkpoint dir (InternVLA-A1.5
#                           only; ignored under the default --action_loss_only).
#   WAN_VAE_PATH          : override for the WAN VAE weights.
#   ACTION_LOSS_ONLY_FLAG : default --action_loss_only (skip WAN loading).
#                           Set to --no-action_loss_only to load the WAN branch.
#   INFERENCE_BACKEND     : default standard. optimized requires
#                           --action_loss_only.
#   STATS_KEY             : default widowx (matches bridge dataset).
#   ROBOT_TYPE            : default widowx (drives image_mapping / action_mode
#                           resolution on the server side).
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJ_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
SERVER_ENV="${SERVER_ENV:-lerobot_lab}"
BASE_PORT="${BASE_PORT:-10086}"
SESSION_NAME="${SESSION_NAME:-policy_servers}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
# WAN paths only matter for InternVLA-A1.5 checkpoints running with
# --no-action_loss_only. The default action_loss_only=True skips WAN loading.
WAN_MODEL_PATH="${WAN_MODEL_PATH:-}"
WAN_VAE_PATH="${WAN_VAE_PATH:-}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
# Bridge dataset is robot_type=widowx; pin both so the server picks the right
# subset out of multi-key stats.json and resolves Control Mode from the
# widowx schema (action_mode=end_effector).
STATS_KEY="${STATS_KEY:-widowx}"
ROBOT_TYPE="${ROBOT_TYPE:-widowx}"

MODEL_PATH="${1:-}"
NUM_PORTS="${2:-8}"
NUM_GPUS="${3:-8}"

if [ -z "${MODEL_PATH}" ]; then
    echo "Usage: $0 <CKPT_PATH> <NUM_PORTS> <NUM_GPUS>"
    exit 1
fi
if ! [[ "${NUM_PORTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_PORTS must be a positive integer"
    exit 1
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer"
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: CKPT_PATH does not exist: ${MODEL_PATH}"
    exit 1
fi

vlm_arg=""
if [ -n "${VLM_MODEL_PATH}" ]; then
    vlm_arg="--vlm_model_path ${VLM_MODEL_PATH}"
fi

wan_arg=""
if [ -n "${WAN_MODEL_PATH}" ]; then
    wan_arg="--wan_model_path ${WAN_MODEL_PATH}"
fi
if [ -n "${WAN_VAE_PATH}" ]; then
    wan_arg="${wan_arg} --wan_vae_path ${WAN_VAE_PATH}"
fi

stats_arg=""
if [ -n "${STATS_KEY}" ]; then
    stats_arg="--stats_key ${STATS_KEY}"
fi
if [ -n "${ROBOT_TYPE}" ]; then
    stats_arg="${stats_arg} --robot_type ${ROBOT_TYPE}"
fi

echo "============================================"
echo "Policy servers (SimplerEnv eval)"
echo "  Project root : ${PROJ_ROOT}"
echo "  Checkpoint   : ${MODEL_PATH}"
echo "  Conda env    : ${SERVER_ENV}"
echo "  Base port    : ${BASE_PORT}"
echo "  Workers      : ${NUM_PORTS} across ${NUM_GPUS} GPU(s)"
echo "  Resize size  : ${RESIZE_SIZE}"
echo "  VLM path     : ${VLM_MODEL_PATH:-<config default>}"
echo "  WAN model    : ${WAN_MODEL_PATH:-<config default (unused under --action_loss_only)>}"
echo "  WAN VAE      : ${WAN_VAE_PATH:-<derived from WAN model dir>}"
echo "  Action loss  : ${ACTION_LOSS_ONLY_FLAG}"
echo "  Inference    : ${INFERENCE_BACKEND}"
echo "  Stats key    : ${STATS_KEY:-<auto>}"
echo "  Robot type   : ${ROBOT_TYPE:-<auto>}"
echo "============================================"

# Reset the tmux session.
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null
tmux new-session -d -s "${SESSION_NAME}"

setup_window() {
    local target=$1
    tmux send-keys -t "${target}" "export TOKENIZERS_PARALLELISM=false" Enter
    tmux send-keys -t "${target}" "source ${CONDA_ROOT}/etc/profile.d/conda.sh" Enter
    tmux send-keys -t "${target}" "conda activate ${SERVER_ENV}" Enter
    tmux send-keys -t "${target}" "export PYTHONPATH=${PROJ_ROOT}:${PROJ_ROOT}/src:\${PYTHONPATH:-}" Enter
}

for ((i=0; i<NUM_PORTS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$((i % NUM_GPUS))
    WIN_NAME="server-$(printf "%02d" ${i})"

    if [ "${i}" -eq 0 ]; then
        tmux rename-window -t "${SESSION_NAME}:0" "${WIN_NAME}"
    else
        tmux new-window -d -t "${SESSION_NAME}" -n "${WIN_NAME}"
    fi

    target="${SESSION_NAME}:${WIN_NAME}"
    setup_window "${target}"
    tmux send-keys -t "${target}" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ${PROJ_ROOT}/evaluation/LIBERO/policy_server/server_policy.py \
            --ckpt_path '${MODEL_PATH}' \
            --port ${PORT} \
            --device cuda \
            --resize_size ${RESIZE_SIZE} \
            --idle_timeout -1 \
            ${ACTION_LOSS_ONLY_FLAG} \
            --inference_backend ${INFERENCE_BACKEND} \
            ${stats_arg} ${vlm_arg} ${wan_arg}" Enter
done

echo "Started ${NUM_PORTS} servers in tmux session '${SESSION_NAME}'"
echo "Ports : $(seq -s ', ' ${BASE_PORT} $((BASE_PORT + NUM_PORTS - 1)))"
echo "GPUs  : 0..$((NUM_GPUS - 1))  (round-robin)"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
