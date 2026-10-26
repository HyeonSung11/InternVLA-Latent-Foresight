#!/usr/bin/env bash
# RoboTwin 평가 — 50과제 × N회 · demo_clean · 단일 GPU (09-21)
#  사용: bash launch/robotwin_eval.sh <TAG> <CKPT_DIR(pretrained_model)> [GPU=1] [EPISODES=20]
#        과제를 좁히려면 TASKS="0 12 44" 처럼 환경변수로.
#  설정은 원논문 evaluation/RoboTwin/eval.sh 기본값과 동일:
#        action abs · infer-horizon 20 · standard backend · seed 42 · instruction unseen · stats aloha
#  결과: outputs/<TAG>_rt_eval/robotwin/demo_clean/<task>/{success_*,failure_*}  (robotwin_result_stats.py 형식)
#  재개: 이미 EPISODES 개 결과가 있는 과제는 건너뜀. 결과가 0개면(시뮬 크래시 등) 1회 재시도.
source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh && conda activate internvla_a1_5
set -uo pipefail
TAG="$1"; CKPT="$(readlink -f "$2")"; GPU="${3:-1}"; EP="${4:-20}"   # 절대 경로 필수: 아래에서 RoboTwin 폴더로 cd 한다
REPO=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=$REPO/outputs/${TAG}_rt_eval; LOGD=$REPO/logs/robotwin/eval_${TAG}; mkdir -p "$ROOT" "$LOGD"
[[ -f "$CKPT/stats.json" && -f "$CKPT/config.json" ]] || { echo "[ERROR] 체크포인트 아님: $CKPT"; exit 2; }
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$REPO/src:$REPO/third_party/RoboTwin:${PYTHONPATH:-}"
cd $REPO/third_party/RoboTwin
mapfile -t NAMES < <(python -c "import sys; sys.path.insert(0,'$REPO/evaluation/RoboTwin'); import inference as m; print('\n'.join(m.TASK_NAMES))" 2>/dev/null)
[[ ${#NAMES[@]} -eq 50 ]] || { echo "[ERROR] 과제 목록 로드 실패 (${#NAMES[@]}개)"; exit 3; }
log(){ echo "[$(date '+%F %T')] $*"; }
count(){ ls "$1" 2>/dev/null | grep -cE '^(success|failure)'; }
log "평가 시작 $TAG · ckpt=$CKPT · GPU$GPU · 과제당 ${EP}회"
for IDX in ${TASKS:-$(seq 0 49)}; do
  NAME=${NAMES[$IDX]}; OUT=$ROOT/robotwin/demo_clean/$NAME
  [[ $(count "$OUT") -ge $EP ]] && { log "[SKIP] $IDX $NAME"; continue; }
  for attempt in 1 2; do
    rm -rf "$OUT"
    PYTHONUNBUFFERED=1 python $REPO/evaluation/RoboTwin/inference.py --ckpt-path "$CKPT" --video-dir "$OUT" \
      --task-config demo_clean --task-idx $IDX --action-mode abs --infer-horizon 20 \
      --inference-backend standard --num-episodes $EP > "$LOGD/${IDX}_${NAME}.log" 2>&1
    rc=$?; n=$(count "$OUT")
    [[ $n -gt 0 ]] && break
    log "  !! $IDX $NAME 결과 0개 (rc=$rc) — 재시도 $attempt"
  done
  s=$(ls "$OUT" 2>/dev/null | grep -c '^success')
  log "$IDX $NAME  성공 $s / $n"
done
python - "$ROOT/robotwin/demo_clean" <<'PY'
import os, sys
root = sys.argv[1]; S = N = 0; rows = []
for t in sorted(os.listdir(root)):
    fs = os.listdir(os.path.join(root, t)); s = sum(f.startswith("success") for f in fs); n = s + sum(f.startswith("failure") for f in fs)
    S += s; N += n; rows.append((t, s, n))
print(f"과제 {len(rows)}개 · 에피소드 {N} · 성공 {S} · 성공률 {100*S/max(N,1):.2f}%")
PY
log "평가 종료 $TAG"
