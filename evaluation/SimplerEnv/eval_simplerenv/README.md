# SimplerEnv  evaluation

## Layout

```
evaluation/SimplerEnv/eval_simplerenv/
├── README.md                ← this file
├── main.py                  ← evaluation client (run inside `simplerenv_state`)
├── env_options/
│   └── bridge.json          ← 576 episodes × 4 task suites
└── scripts/
    ├── deploy.sh            ← N policy servers in a tmux session
    └── launch_simplerenv.sh ← N parallel eval clients
```

## Environment Setup

| Component               | Conda env          | Notes                                                         |
| ----------------------- | ------------------ | ------------------------------------------------------------- |
| Policy server           | `lerobot_lab`      | Runs `evaluation/LIBERO/policy_server/server_policy.py`.       |
| SimplerEnv eval client  | `simplerenv_state` | SAPIEN + ManiSkill2_real2sim from `allenzren/SimplerEnv`.     |


```bash
# Create and activate a new conda environment
conda create -n simplerenv_state python=3.10 -y
conda activate simplerenv_state

# Clone the SimplerEnv repository
git clone https://github.com/allenzren/SimplerEnv.git --recurse-submodules

# Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install ManiSkill2 real-to-sim environments and their dependencies:
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .

# Install SimplerEnv
cd ..
pip install -e .

pip install numpy==1.24.4 tyro transformers==4.57.1 pandas
```


## Quick start

```bash
# Terminal 1 — start 8 servers across 8 GPUs (tmux session: policy_servers)
# InternVLA-A1.5 checkpoints load fine under the default --action_loss_only;
# only pass WAN_MODEL_PATH when running with --no-action_loss_only.
VLM_MODEL_PATH=/path/to/Qwen3.5-2B-Action \
bash evaluation/SimplerEnv/eval_simplerenv/scripts/deploy.sh \
    /path/to/pretrained_model 8 8

# Terminal 2 — wait for the servers to print "server running ...", then:
SimplerEnv_PATH=/path/to/allenzren/SimplerEnv \
bash evaluation/SimplerEnv/eval_simplerenv/scripts/launch_simplerenv.sh \
    8 bridge ./logs
```

`deploy.sh` defaults `STATS_KEY=widowx` and `ROBOT_TYPE=widowx` so the server
picks the bridge subset out of a multi-key `stats.json` and the widowx
image-mapping (single `observation.images.image_0` → slot 0) drives input
validation. Pass `STATS_KEY=` (empty) if your checkpoint only has one stats
key and you want the server to auto-pick.

Each rank `i` connects to `BASE_PORT + i` (default `BASE_PORT=10086`),
processes its slice of `env_options/bridge.json`, and writes a per-rank CSV
under `<LOG_PATH>/bridge/results/{i}.csv`. Rank 0 aggregates the per-rank CSVs
into `all_results.csv` once every rank has dropped its `*.end` marker.

Per-episode rollout videos land under `<LOG_PATH>/bridge/videos/<task_suite>/`.
