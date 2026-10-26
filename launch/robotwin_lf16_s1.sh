#!/usr/bin/env bash
# 10.10 RoboTwin LF v2 두 번째 시드: rt_lf16_s1 = rt_lf16_s42 와 동일 레시피(LF v2 · 배치 2×누적 8=16 · 120k · abs · foresight 동결) + SEED=1.
#   목적: RoboTwin 시드 분산 측정 (c11·mv 판정의 잡음 기준). GPU0 학습 → mv 체인 평가가 끝난 뒤 GPU1 에서 50과제×20회 평가.
source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh && conda activate internvla_a1_5
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
log(){ echo "[$(date '+%F %T')] $*"; }
gpu_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | awk '{exit !($1<1000)}'; }
TAG=rt_lf16_s1; STEPS=120000
if [[ ! -d outputs/robotwin_train/$TAG/checkpoints/$STEPS/pretrained_model ]]; then
  until gpu_free 0; do sleep 120; done
  log "GPU0: 학습 시작 $TAG (seed 1)"
  env GPUS=0 BATCH=2 REAL_GRAD_ACCUM=8 GRAD_ACCUM=1 STEPS=$STEPS SAVE_FREQ=40000 SEED=1 PORT=6998 TAG=$TAG \
      LATENT_FORESIGHT=true LF_TARGET=dino LF_LOSS=cosine \
      bash launch/robotwin_train.sh > logs/robotwin/train_$TAG.log 2>&1
  log "학습 프로세스 종료 rc=$? (완료 판정은 체크포인트로)"
  grep -oE "\[BATCH\].*" logs/robotwin/train_$TAG.log | head -1 | sed 's/^/    /'
fi
[[ -d outputs/robotwin_train/$TAG/checkpoints/$STEPS/pretrained_model ]] || { log "!!! 체크포인트 없음 — 중단"; exit 1; }
log "GPU1 대기 (mv 체인의 RoboTwin 평가가 끝날 때까지)"
until grep -q "체인 완료" logs/robotwin/chain_rt_sp16_mv.log 2>/dev/null && ! pgrep -f "robotwin_eval.sh" >/dev/null \
      && [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) -lt 3000 ]]; do sleep 120; done
log "GPU1: RoboTwin 평가 시작 $TAG (50과제 × 20회)"
bash launch/robotwin_eval.sh $TAG outputs/robotwin_train/$TAG/checkpoints/$STEPS/pretrained_model 1 20 > logs/robotwin/eval_$TAG.log 2>&1
log "평가 종료: $(tail -1 logs/robotwin/eval_$TAG.log)"
python3 - <<'PY' | tee -a logs/robotwin/final_summary.txt
import os
for tag in ['rt_wan16_s42','rt_lf16_s42','rt_lf16_s1','rt_sp16_c11','rt_sp16_mv']:
    d=f'outputs/{tag}_rt_eval/robotwin/demo_clean'
    if not os.path.isdir(d): print(f'[rt_lf16_s1 후] {tag}: 결과 없음'); continue
    s=n=0
    for t in sorted(os.listdir(d)):
        fs=os.listdir(os.path.join(d,t)); s+=sum(f.startswith('success') for f in fs); n+=sum(f.startswith(('success','failure')) for f in fs)
    print(f'[rt_lf16_s1 후] {tag}: {s}/{n} = {100*s/max(n,1):.2f}%')
PY
log "===== RoboTwin LF v2 seed 1 체인 완료 ====="
