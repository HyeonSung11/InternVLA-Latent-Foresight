#!/usr/bin/env bash
# 9.29 기하 쿼리 토큰(Spatial Query Tokens) 학습 큐 — GPU0 전용, 순차.
#  기준선: b16_ctrl (LF v2 · 유효배치 16 = 4×누적4 · 100k iter · seed 42 · GPU0) = 96.35
#  교정 런 sp16_vggt_c11: 타깃 중심화(이미지별 패치 평균 제거) + VGGT 11층 — 마지막층은 패치 간 공통 방향 96%라 상수 예측으로도 cosine 손실 0.025 (퇴화)
#  얼린 VGGT 교사는 원본 WAN 과 같이 GPU1 에 둔다(GPU0 에 두면 paged optimizer 페이징으로 5초/step). 학습 모델은 GPU0.
#  끝나면 logs/alw/ready_for_eval.txt 에 태그를 적어 GPU1 평가 워커가 집어가게 한다.
source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh && conda activate internvla_a1_5
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
mkdir -p logs/sp logs/alw
M=logs/alw/manifest.tsv
[[ -f $M ]] || printf "tag\tgpu\tbatch\treal_accum\teff\tsteps\tstart\tenv\n" > $M
log(){ echo "[$(date '+%F %T')] $*"; }
gpu_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{exit !($1<1000)}'; }

train(){  # tag steps real_accum [env...]
  local tag=$1 st=$2 ra=$3; shift 3
  [[ -d outputs/full_train/$tag/checkpoints/$st ]] && { log "[SKIP] $tag 완료됨"; return 0; }
  until gpu_free; do sleep 60; done; sleep 20
  log "START $tag (GPU0 · 배치 $((4*ra)) · $st step) : ${*:-플래그 없음}"
  printf "%s\t0\t4\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$ra" "$((4*ra))" "$st" "$(date '+%F %T')" "${*:-none}" >> $M
  env GPUS=0,1 STEPS=$st BATCH=4 GRAD_ACCUM=1 REAL_GRAD_ACCUM=$ra SAVE_FREQ=$st SEED=42 PORT=6983 TAG=$tag \
      LATENT_FORESIGHT=true LF_TARGET=dino LF_LOSS=cosine "$@" \
      bash launch/full_train.sh > "logs/sp/train_${tag}.log" 2>&1
  local rc=$?
  grep -oE "\[BATCH\].*" "logs/sp/train_${tag}.log" | tail -1 | sed 's/^/    /'
  grep -oE "\[SPATIAL\].*" "logs/sp/train_${tag}.log" | tail -1 | sed 's/^/    /'
  log "END $tag rc=$rc"
  [[ $rc -ne 0 ]] && { log "!!! $tag 실패 — 큐 중단"; exit 1; }
  echo "$tag" >> logs/alw/ready_for_eval.txt      # GPU1 평가 워커가 집어간다 (최신 ckpt step 자동 선택)
}

train sp16_vggt_c11 100000 4 USE_SPATIAL=true SP_TEACHER=vggt SP_DEVICE=cuda:1 SP_TOKENS=64 SP_WEIGHT=0.15 SP_WARMUP=1000 SP_ISOLATE=true SP_CENTER=true SP_LAYER=11
log "===== 9.29 교정 런(중심화·11층) 큐 완료 ====="
