# Installation

## Requirements

The code is built and tested with **Python 3.11**, **CUDA 12.8**, and **PyTorch 2.10.0**.

## Preparation

### 1. Clone the repository

```bash
git clone https://github.com/InternRobotics/InternVLA-A-series.git
cd InternVLA-A1.5
```

### 2. Create Conda environment

```bash
conda create -y -n internvla_a1_5 python=3.11
conda activate internvla_a1_5
pip install --upgrade pip
```

### 3. Install system dependencies

We use FFmpeg for video encoding/decoding and SVT-AV1 for efficient storage.

```bash
conda install -c conda-forge ffmpeg svt-av1 -y
```

### 4. Install PyTorch for CUDA 12.8

```bash
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install Python dependencies

```bash
pip install transformers==5.2.0
pip install -e .
```

### 6. Install attention kernels

```bash
pip install flash-attn==2.8.3 flash-linear-attention==0.5.0 causal-conv1d==1.6.1 \
  --no-build-isolation
```

If the installation fails on your CUDA/PyTorch platform, follow the upstream build instructions and install the matching versions from [flash-attn](https://github.com/Dao-AILab/flash-attention), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d).

### 7. Patch HuggingFace Transformers

InternVLA-A1.5 uses custom model code for Qwen3.5 and robot-learning policies. Copy the replacement modules into the installed Transformers package:

```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.11/site-packages/transformers/

cp -r src/lerobot/policies/pi0/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/pi05/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/internvla_a1_5/transformers_replace/models ${TRANSFORMERS_DIR}
```

Make sure `${TRANSFORMERS_DIR}` exists before copying.

### 8. Configure environment variables

```bash
export HF_TOKEN=your_token
export HF_HOME=path_to_huggingface
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
```

### 9. Optional: download VGM weights

If you need to train with the VGM (video generation model) branch enabled, download **Wan2.2-TI2V-5B** to `${HF_HOME}/hub`:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir ${HF_HOME}/hub/Wan2.2-TI2V-5B
```

This step can be skipped for action-only inference/evaluation or runs that disable video foresight supervision.

### 10. Link local LeRobot cache

If your datasets are stored under `${HF_HOME}/lerobot`, link them into this repository:

```bash
ln -s ${HF_HOME}/lerobot data
```

This allows the training scripts to access datasets through `./data/`.
