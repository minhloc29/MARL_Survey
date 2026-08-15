# MATE

Multi-Agent Tracking Environment. Two training pipelines are provided: a PPO
goal-mask trainer ([train.py](train.py)) and a standalone MAC-Flow
(flow-matching CTDE) trainer ([train_flow.py](train_flow.py)).

## Installation

1. Install [uv](https://github.com/astral-sh/uv):

    ```bash
    pip install uv
    or
    pip3 install uv
    ```

4. Install or sync dependencies with `uv sync`:

    ```bash
    uv sync
    ```

3. Activate the environment:

    - On Unix/macOS:

      ```bash
      source .venv/bin/activate
      ```

    - On Windows:

      ```cmd
      .venv\Scripts\activate
      ```

You're now ready to start using MATE!

---

## Training: train.py (PPO goal-mask)

PPO over the multi-agent goal-selection action space, coordinated by
`GoalsBaseAgent` and the attention-based `Net`/`Critic` networks.

```bash
# Train with defaults
python train.py

# Run on 16 parallel envs, 2048 steps each per iteration
python train.py --num-envs 16 --n-steps 2048 --total-timesteps 10000000

# Train on GPU
python train.py --device auto        # cuda, mps (Apple Silicon), or cpu

# Tune learning rates
python train.py --actor-lr 5e-5 --critic-lr 1e-4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--actor-lr` | `5e-5` | Actor (policy) learning rate |
| `--critic-lr` | `1e-4` | Critic (value) learning rate |
| `--device` | `auto` | `auto`, `cuda`, `mps` (Apple Silicon), or `cpu` |
| `--num-envs` | `16` | Number of parallel environments |
| `--n-steps` | `2048` | Steps per env per iteration |
| `--batch-size` | `128` | Training batch size |
| `--total-timesteps` | `10_000_000` | Total timesteps to train |
| `--save-every` | `10_000` | Save agent every N timesteps |

W&B logging is on by default via the API key embedded in
[train.py](train.py) (see the `wandb_api=` argument to `agent.fit()`). It logs
`rollout/score`, `rollout/avg_score`, plus **`rollout/coverage`** and
**`rollout/avg_coverage`** (mean per-episode camera coverage).

---

## Training: train_flow.py (MAC-Flow)

Standalone flow-matching CTDE trainer with its own replay buffer. It drives
each camera with low-level continuous actions and runs three training stages
jointly: (1) advantage-weighted flow-matching BC, (2) per-agent IGM critics
with TD, (3) one-step-policy distillation from the joint flow.

```bash
# Train and save checkpoints every 1000 steps
python train_flow.py train --steps 10000000 

# Train on GPU (MPS on Apple Silicon)
python train_flow.py train --steps 5000 --device mps

# Resume training from a checkpoint
python train_flow.py train --steps 5000 --resume ckpts/step_3000.pt

# Log curves (losses + coverage) to W&B
python train_flow.py train --steps 5000 --wandb-api <YOUR_WANDB_KEY>

# Evaluate a checkpoint over 20 episodes
python train_flow.py evaluate --checkpoint ckpts/step_5000.pt --episodes 20

# Render a rollout from a checkpoint
python train_flow.py render --checkpoint ckpts/step_5000.pt
```

### Subcommands

| Command | Purpose |
|---------|---------|
| `train` | Train and periodically save checkpoints |
| `evaluate` | Load a checkpoint, run the learned one-step policies, report stats |
| `render` | Load a checkpoint and play one rollout with pygame rendering |

### `train` options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `MATE-4v8-9.yaml` | MATE config file |
| `--embed-dim` | `128` | Embedding dim for the observation encoders |
| `--act-dim` | `2` | Per-camera action dim (continuous) |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `--steps` | `5000` | Total training steps |
| `--warmup` | `1000` | Replay-buffer warmup steps before updates start |
| `--batch-size` | `64` | Update batch size |
| `--replay-capacity` | `100000` | Replay buffer capacity |
| `--lr` | `3e-4` | Adam learning rate (all networks) |
| `--gamma` | `0.99` | Return discount factor |
| `--tau` | `0.005` | Target-critic Polyak coefficient |
| `--alpha` | `3.0` | Distillation regularizer weight (Eq. 9) |
| `--adv-beta` | `1.0` | Advantage temperature for advantage-weighted flow-BC |
| `--flow-steps` | `10` | Euler ODE steps when sampling the joint flow |
| `--log-every` | `200` | Print/wandb log interval in steps |
| `--save-every` | `1000` | Checkpoint interval in steps |
| `--outdir` | `ckpts` | Directory for checkpoints |
| `--resume` | — | Path to a checkpoint to resume from |
| `--wandb-api` | — | If set, log curves (losses + coverage) to W&B |
| `--seed` | `0` | Random seed |

### `evaluate` / `render` options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `MATE-4v8-9.yaml` | MATE config file |
| `--embed-dim` | `128` | Embedding dim (must match the checkpoint) |
| `--act-dim` | `2` | Action dim (must match the checkpoint) |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `--checkpoint` | *(required)* | Path to the checkpoint to load |
| `--episodes` | `20` | *(evaluate)* number of episodes |
| `--max-steps` | `2000` | *(render)* max rollout steps |

### Coverage logging

Both trainers read MATE's per-camera `coverage_rate` from `info['infos']` and
report it separately from raw reward:

- **train.py (wandb):** `rollout/coverage`, `rollout/avg_coverage`.
- **train_flow.py (stdout + wandb):** `last_ep_cov` line, and a `coverage_rate`
  curve when `--wandb-api` is set.



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


python train.py --algo mappo --steps 10000000 