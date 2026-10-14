# MATE

Multi-Agent Tracking Environment with two training pipelines, both exposed
through [`train.py`](train.py): an on-policy PPO/MAPPO goal-mask trainer and a
unified `--algo` dispatcher for a suite of MARL algorithms on the discrete
camera-team task (`mate/algorithms/`).

## Codebase at a glance

- **`train.py`** — entry point. `python train.py --algo <name>` routes to a
  single algorithm; each algorithm has its own `--<algo>-*` flags.
- **`mate/`** — the environment (`mate/environment.py`), wrappers, and the
  algorithm library under `mate/algorithms/`.
- **`mate/algorithms/`** — self-contained MARL implementations (numpy+torch),
  each with a `<Name>Config` + `<Name>Trainer` + `run_<name>()`.
- **`agents/`, `net.py`, `common_net/`** — heuristic agents and shared
  attention/encoder networks for the goal-mask trainer.
- **`train_flow.py`** — standalone MAC-Flow (flow-matching CTDE) trainer.
- **`gym_agent/`, `flow/`** — reusable RL/flow support libraries.

Available algorithms via `--algo`: `ppo`, `mappo`, `maddpq`, `qmix`, `qplex`,
`rode`, `coma`, `mat`, `dcg`, `i2c`, **`dm2`**, **`facmac`**.

## Quick train

```bash
# Default on-policy goal-mask PPO
python train.py

# MAPPO
python train.py --algo mappo --total-timesteps 10000000

# Discrete MARL algorithms (camera team)
python train.py --algo qmix
python train.py --algo coma
python train.py --algo dm2 --dm2-rew-type mixed --dm2-gail-sil --dm2-update-gail
python train.py --algo facmac

# Device control
python train.py --algo mappo --device auto   # cuda, mps (Apple Silicon), or cpu
```

For per-algorithm options (learning rates, epsilon, batch size, target-update
cadence, etc.) run `python train.py --help` or see the `--<algo>-*` flags.

### Shared training budget (default 10M)

Every algorithm — PPO/MAPPO **and** the `--algo` suite — trains for the same
number of **policy actions**, controlled by one flag:

```bash
python train.py --algo mappo   --total-timesteps 10000000  # on-policy PPO/MAPPO
python train.py --algo qmix    --total-timesteps 10000000  # off-policy value decomposition
python train.py --algo coma    --total-timesteps 10000000  # episode-based off-policy
python train.py --algo mat     --total-timesteps 10000000  # deterministic (updates = steps / rollout)
```

`--total-timesteps` (default `10_000_000`) is the single shared budget for all 12
algorithms, so their wandb curves are comparable. 1 step = 1 RL policy action;
PPO/MAPPO additionally apply frame-skip (8 env transitions per action) as part of
their hierarchical design.

The per-algorithm count flags (`--steps`, `--<algo>-episodes`, `--mat-updates`)
default to `None` and are derived from `--total-timesteps`. Episode-based algos
(rode, coma, dcg, i2c, dm2, facmac) run under a live env-step accumulator and stop
at the budget even though they terminate episodes early on `done`; mat derives
`updates = total_timesteps // rollout_steps` exactly. You may still override any
count flag explicitly, e.g. `--algo mat --mat-updates 1000`.

Every algorithm logs rewards and losses to **W&B** when an API key is provided
(`--wandb-api <key>`, or the `WANDB_API_KEY` env var), one run per algorithm;
training still runs normally without one.

## Installation

1. Install [uv](https://github.com/astral-sh/uv): `pip install uv`
2. `uv sync`
3. `source .venv/bin/activate`

---

## Training: train.py (PPO goal-mask)

PPO over the multi-agent goal-selection action space, coordinated by
`GoalsBaseAgent` and the attention-based `Net`/`Critic` networks.

```bash
python train.py --num-envs 16 --n-steps 2048 --total-timesteps 10000000
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

W&B logging is on by default via the `wandb_api=` argument to `agent.fit()`. It
logs `rollout/score`, `rollout/avg_score`, plus **`rollout/coverage`** and
**`rollout/avg_coverage`** (mean per-episode camera coverage).

---

## Training: train_flow.py (MAC-Flow)

Standalone flow-matching CTDE trainer with its own replay buffer. It drives
each camera with low-level continuous actions and runs three training stages
jointly: (1) advantage-weighted flow-matching BC, (2) per-agent IGM critics
with TD, (3) one-step-policy distillation from the joint flow.

```bash
python train_flow.py train --steps 10000000
python train_flow.py train --steps 5000 --device mps
python train_flow.py train --steps 5000 --resume ckpts/step_3000.pt
python train_flow.py evaluate --checkpoint ckpts/step_5000.pt --episodes 20
python train_flow.py render --checkpoint ckpts/step_5000.pt
```

### Subcommands

| Command | Purpose |
|---------|---------|
| `train` | Train and periodically save checkpoints |
| `evaluate` | Load a checkpoint, run the learned one-step policies, report stats |
| `render` | Load a checkpoint and play one rollout with pygame rendering |

### Coverage logging

Both trainers read MATE's per-camera `coverage_rate` from `info['infos']` and
report it separately from raw reward:

- **train.py (wandb):** `rollout/coverage`, `rollout/avg_coverage`.
- **train_flow.py (stdout + wandb):** `last_ep_cov` line, and a `coverage_rate`
  curve when `--wandb-api` is set.
