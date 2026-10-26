#!/usr/bin/env bash
# 학습 전담 큐 — 전부 GPU0 (09-15)
#
#  원칙: 논문에 실리는 모든 학습은 GPU0 한 장에서만 수행한다.
#        (조건 간 비교에 하드웨어 변수가 끼어들 여지를 아예 없앤다.)
#        평가는 학습이 끝난 체크포인트로 추론만 하므로 GPU1 의 eval_worker_g1.sh 가 병렬 처리한다.
#
#  배치 4 관련 (09-15 발견):
#   accelerate launch --gradient_accumulation_steps 는 DeepSpeed 사용 시에만 반영되고(공식 도움말),
#   update_policy 는 accelerator.accumulate() 를 쓰지 않아 GRAD_ACCUM 이 no-op 이었다.
#   이 저장소의 모든 기존 런은 배치 4 였다(저장 설정 39개 전수 확인).
#   실제 누적이 필요하면 REAL_GRAD_ACCUM 을 쓴다(09-15 구현, 기본 1 은 기존과 비트 동일 확인).
#
#  기준선은 새로 돌리지 않는다. GRAD_ACCUM 이 no-op 이었음이 밝혀지면서 기존
#  lf_v2_iso16(seed 42) 의 설정이 이 큐와 완전히 동일해졌기 때문이다(배치 4 · 25k · LF v2 · seed 42).
#    기준: lf_v2_iso16 s42 = 83.95 · 5시드 평균 83.58 ± 0.82
#  코드 수정(accumulate 래퍼·[BATCH] 로그)이 기존 동작을 바꾸지 않음은 실측으로 확인하였다:
#    수정 전/후 step50 에서 loss 8.047 · grdn 111.538 로 일치.
#
#  순서 (런당 ~9.4h, 마지막만 ~37.5h):
#    1 b4_all      제안 1+2a+3 전부 켬 — 스크리닝. 효과가 있는지 먼저 본다
#    2 b4_tgate    제안 1 우리 기여 (t-게이트 운동 임계도 가중)
#    3 b4_strat    제안 2a 층화 시간 샘플링
#    4 b4_sobolev  제안 3 Sobolev 손실
#    5 b4_inv      제안 1 원논문형 = 게이트 유무 대조군
#    6 b16_ctrl    배치 16 (REAL_GRAD_ACCUM=4, 100k step 으로 갱신 25k 유지) — 배치 효과 측정
source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh && conda activate internvla_a1_5
set -uo pipefail
cd ${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
mkdir -p logs/alw
M=logs/alw/manifest.tsv
[[ -f $M ]] || printf "tag\tgpu\tbatch\treal_accum\teff\tsteps\tstart\tenv\n" > $M
log(){ echo "[$(date '+%F %T')] $*"; }
gpu_free(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{exit !($1<1000)}'; }

train(){  # tag steps real_accum [env...]
  local tag=$1 st=$2 ra=$3; shift 3
  [[ -d outputs/full_train/$tag/checkpoints/0$st ]] && { log "[SKIP] $tag 완료됨"; return 0; }
  until gpu_free; do sleep 60; done; sleep 20
  log "START $tag (GPU0 · 배치 $((4*ra)) · $st step) : ${*:-플래그 없음}"
  printf "%s\t0\t4\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$ra" "$((4*ra))" "$st" "$(date '+%F %T')" "${*:-none}" >> $M
  env GPUS=0 STEPS=$st BATCH=4 GRAD_ACCUM=1 REAL_GRAD_ACCUM=$ra SAVE_FREQ=$st SEED=42 PORT=6980 TAG=$tag \
      LATENT_FORESIGHT=true LF_TARGET=dino LF_LOSS=cosine "$@" \
      bash launch/full_train.sh > "logs/alw/train_${tag}.log" 2>&1
  local rc=$?
  grep -oE "\[BATCH\].*" "logs/alw/train_${tag}.log" | tail -1 | sed 's/^/    /'
  log "END $tag rc=$rc"
  [[ $rc -ne 0 ]] && { log "!!! $tag 실패 — 큐 중단"; exit 1; }
  echo "$tag" >> logs/alw/ready_for_eval.txt      # GPU1 평가 워커가 집어간다
}

# 1) 스크리닝 — 세 제안을 전부 켠 조합
train b4_all     25000 1 ALW=inv_speed_tgate ALW_FN=inv ALW_CLIP=3.0 ALW_DIMS=6 STRATIFIED=true SOBOLEV=0.1
# 2~5) 개별 분해 — 조합에서 효과가 보이면 어느 성분 때문인지 가른다
train b4_tgate   25000 1 ALW=inv_speed_tgate ALW_FN=inv ALW_CLIP=3.0 ALW_DIMS=6
train b4_strat   25000 1 STRATIFIED=true
train b4_sobolev 25000 1 SOBOLEV=0.1
train b4_inv     25000 1 ALW=inv_speed       ALW_FN=inv ALW_CLIP=3.0 ALW_DIMS=6
# 6) 별도 질문 — 배치를 키우면 오르는가
train b16_ctrl  100000 4
log "===== GPU0 학습 큐 완료 ====="
