#!/usr/bin/env bash
# RoboTwin 기준선 — 논문 원모델(WAN 경로)을 우리 조건으로 학습 (09-21)
#
#  레시피: 원논문 internvla_a15_finetune_robotwin.sh 와 동일
#          (60k step · lr 5e-5 · warmup 2000 · decay 5e-6 · chunk 50 · foresight 토큰 동결 · aloha 외부 통계)
#  차이:   배치 4 (LIBERO 모든 런과 같은 조건, 원논문은 8 × 다중 GPU) · gradient checkpointing(메모리용, 계산 동일)
#  배치:   GRAD_ACCUM 은 no-op 이므로 1, REAL_GRAD_ACCUM 미지정(=1) → 실제 유효배치 4 ([BATCH] 로그로 확인)
#  GPU:    학습 GPU0 · WAN(얼린 5B) GPU1. WAN 조건은 원래 두 장을 쓴다.
#  순서:   GPU 두 장이 모두 빌 때까지 대기(GPU1 은 b16_ctrl 평가 중) → 60 step 점검 → 본 학습
source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh && conda activate internvla_a1_5
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
mkdir -p logs/robotwin
log(){ echo "[$(date '+%F %T')] $*"; }
gpu_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | awk '{exit !($1<1000)}'; }
COMMON="GPUS=0,1 WAN_DEVICE=cuda:1 BATCH=4 GRAD_ACCUM=1 SEED=42 FREEZE_LT=true"

log "대기 — GPU0·GPU1 모두 빌 때까지"
until gpu_free 0 && gpu_free 1; do sleep 120; done; sleep 60
log "b16_ctrl 평가 결과: $(ls outputs/b16_ctrl_eval50/*/results_*.json 2>/dev/null | wc -l)/4 suite"

# ---- 1) 60 step 점검: 메모리·속도 확인 ----
log "점검 시작 (60 step)"
( peak0=0; peak1=0
  while :; do read a b < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')
    [ "${a:-0}" -gt "$peak0" ] && peak0=$a; [ "${b:-0}" -gt "$peak1" ] && peak1=$b
    echo "$peak0 $peak1" > logs/robotwin/rt_wan_probe.peak; sleep 3; done ) & SAMPLER=$!
rm -rf outputs/robotwin_train/rt_wan_probe
env $COMMON STEPS=60 PORT=6995 TAG=rt_wan_probe bash launch/robotwin_train.sh > logs/robotwin/rt_wan_probe.log 2>&1
rc=$?; kill $SAMPLER 2>/dev/null
read p0 p1 < logs/robotwin/rt_wan_probe.peak
log "점검 종료 rc=$rc · peak GPU0 ${p0}MiB / GPU1 ${p1}MiB (한계 24576)"
grep -oE "\[BATCH\].*" logs/robotwin/rt_wan_probe.log | tail -1 | sed 's/^/    /'
grep -E "step:50\.0" logs/robotwin/rt_wan_probe.log | grep -oE "loss:[0-9.]+|loss_action:[0-9.]+|updt_s:[0-9.]+" | tr '\n' ' ' | sed 's/^/    /'; echo
if [[ $rc -ne 0 ]] || grep -qiE "out of memory" logs/robotwin/rt_wan_probe.log || ! grep -qE "step:50\.0" logs/robotwin/rt_wan_probe.log; then
  log "!!! 점검 실패 — 본 학습을 시작하지 않음"; exit 1
fi
rm -rf outputs/robotwin_train/rt_wan_probe

# ---- 2) 본 학습 ----
log "본 학습 시작 rt_wan_s42 (60k · 20k 마다 저장)"
printf "rt_wan_s42\tGPU0+WAN@GPU1\t4\t1\t4\t60000\t%s\tWAN 기준선\n" "$(date '+%F %T')" >> logs/robotwin/manifest.tsv
env $COMMON STEPS=60000 SAVE_FREQ=20000 PORT=6996 TAG=rt_wan_s42 bash launch/robotwin_train.sh > logs/robotwin/train_rt_wan_s42.log 2>&1
log "본 학습 종료 rc=$?"
