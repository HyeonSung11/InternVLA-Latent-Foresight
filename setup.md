# Setup

Tested on Ubuntu 22.04, CUDA 12.x, two NVIDIA RTX 3090 (24 GB). The trainable model always runs on GPU 0; GPU 1 is used only for the frozen WAN2.2-5B (original baseline) or the frozen VGGT teacher (geometry-token comparison). Latent Foresight runs need GPU 0 only.

The environment follows the original [InternVLA-A1.5](https://github.com/InternRobotics/InternVLA-A-series) instructions (see `docs/README_InternVLA-A1.5.md`), plus `bitsandbytes` for the 8-bit paged AdamW.

## 1. Environment

```bash
conda create -n internvla_a1_5 python=3.10 -y
conda activate internvla_a1_5
pip install -e .            # LeRobot-based training code in src/
pip install bitsandbytes    # 8-bit paged AdamW
```

Download the public checkpoints (InternVLA-A1.5-base, Qwen3.5-2B, WAN2.2-5B for the baseline) as described in the original README. The launch scripts read them from `$HF_HOME/ckpts/...`; set `HF_HOME` to your Hugging Face cache.

## 2. Data

LIBERO and RoboTwin 2.0 datasets in LeRobot format under `data/` (git-ignored):

```
data/libero/libero_{spatial,object,goal,10}_no_noops_lerobot/
data/robotwin/<task>/aloha-agilex*/
```

RoboTwin 2.0 itself is a submodule in the upstream repo (`third_party/RoboTwin`); clone it there for RoboTwin evaluation. LIBERO evaluation expects the LIBERO repo at `$LIBERO_HOME`.

## 3. Environment variables used by the launch scripts

| Variable | Default | Meaning |
|---|---|---|
| `REPO_ROOT` | script's parent dir | repository root |
| `HF_HOME` | `~/.cache/huggingface` | checkpoints and HF cache |
| `CONDA_ROOT` | `~/miniconda3` | conda installation |
| `LIBERO_HOME` | `~/LIBERO` | LIBERO repo for evaluation |
| `GPUS` / `WAN_DEVICE` | `0` / empty | visible GPUs; `WAN_DEVICE=cuda:1` puts the frozen WAN on the auxiliary GPU |
| `BATCH`, `REAL_GRAD_ACCUM` | `1`, `1` | micro-batch and real gradient accumulation (effective batch = product) |
| `STEPS`, `SEED`, `TAG` | – | run length, seed, output tag |
| `LATENT_FORESIGHT`, `LF_TARGET`, `LF_LOSS` | `false`, `pixel`, `mse` | enable Latent Foresight; `LF_TARGET=dino LF_LOSS=cosine` is v2 (paper) |
| `USE_SPATIAL`, `SP_TEACHER`, `SP_LAYER`, `SP_CENTER`, `SP_DEVICE` | `false`, `vggt`, `-1`, `false`, empty | geometry query tokens (Table 6) |

## 4. Reproducing the paper

```bash
# (a) Original WAN objective, lightweight porting: trainable model on GPU 0, frozen WAN on GPU 1
GPUS=0,1 WAN_DEVICE=cuda:1 BATCH=4 STEPS=100000 SEED=42 TAG=wan_b4 bash launch/full_train.sh

# (b) Latent Foresight v2 (DINOv2 features, cosine loss), single GPU
GPUS=0 BATCH=4 STEPS=100000 SEED=42 TAG=lf_v2_b4 \
  LATENT_FORESIGHT=true LF_TARGET=dino LF_LOSS=cosine bash launch/full_train.sh

# (c) Effective batch 16 control (Table 5)
GPUS=0 BATCH=4 REAL_GRAD_ACCUM=4 STEPS=100000 SEED=42 TAG=lf_v2_b16 \
  LATENT_FORESIGHT=true LF_TARGET=dino LF_LOSS=cosine bash launch/full_train.sh

# (d) Geometry query tokens on top of (c) (Table 6, c11 row) — frozen VGGT on GPU 1
bash launch/train_sp_g0_c11.sh

# LIBERO evaluation (50 trials/task, all four suites)
bash launch/eval50_suites.sh <TAG> <GPU_IDS> outputs/full_train <PORT> spatial object goal long

# RoboTwin 2.0: training and evaluation
bash launch/robotwin_wan_baseline.sh                      # WAN baseline
GPUS=0 BATCH=2 REAL_GRAD_ACCUM=8 STEPS=120000 SEED=42 TAG=rt_lf16 \
  LATENT_FORESIGHT=true LF_TARGET=dino LF_LOSS=cosine bash launch/robotwin_train.sh
bash launch/robotwin_eval.sh <TAG> <ckpt_dir>/pretrained_model 1 20
```

`launch/lora_exp.sh` and `launch/seam_exp.sh` reproduce the frozen / LoRA rows of Table 5.

## Where the changes are

| File | Change |
|---|---|
| `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | Latent Foresight decoder (v1 pixel grid, v2 frozen DINOv2 target, cosine / MSE / InfoNCE), `wan_device` placement of the frozen WAN, geometry query tokens with frozen VGGT teacher and attention isolation, VLM freeze / LoRA / seam-grounding options |
| `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` | All new options (`use_latent_foresight`, `latent_foresight_target`, `use_spatial_tokens`, `spatial_*`, `wan_device`, `freeze_vlm`, `use_vlm_lora`, ...) |
| `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` | Future-frame / multi-view inputs for the training-only teachers |
| `src/lerobot/optim/optimizers.py` | 8-bit paged AdamW (bitsandbytes) and checkpoint save / load compatibility |
| `src/lerobot/scripts/lerobot_train.py` | Real gradient accumulation via `accelerator.accumulate`, effective-batch logging, DDP options, extra loss metrics |
| `evaluation/LIBERO/*` | Suite-wise server-client evaluation (`run_eval_libero_server_client_suites.sh`), per-suite stats / robot type |
| `launch/*.sh` | Training and evaluation scripts used for the paper (see §4) |
