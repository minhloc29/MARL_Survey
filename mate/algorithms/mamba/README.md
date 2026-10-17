# MAMBA for MATE

This integration adapts [jbr-ai-labs/mamba](https://github.com/jbr-ai-labs/mamba)
at commit `2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f` (MIT, see [LICENSE](LICENSE)).
The upstream checkout is at `D:\mamba`; this package is self-contained and does
not require that checkout on `PYTHONPATH`.

## Implementation

`core/` preserves the upstream categorical RSSM with attention between agents,
observation encoder/decoder, reward and continuation predictors, action
information loss, balanced KL, per-agent actor and attention critic. The learner
fits the world model on real sequences, imagines trajectories from posterior
states, computes lambda returns and trains actor/critic using clipped PPO.

The MATE boundary uses rescaled per-camera observations and a `levels x levels`
discrete camera action grid against `GreedyTargetAgent`. Rewards are MATE camera
team rewards, repeated across agents for world-model training. Coverage is logged
separately. All camera actions are available, so the optional availability
predictor is disabled. Online latent inference uses joint camera histories,
including the upstream cross-agent attention; this is not a strictly isolated
per-camera controller.

Adaptations relative to upstream:

- Python 3.13 / modern PyTorch and Gymnasium, without Ray, Flatland or SMAC.
- Local imports; optional W&B only at the runner boundary.
- Episode replay with `T+1` observations and `T` transitions, correctly shifted
  previous actions/rewards and no sequences crossing episode resets.
- Time-limit truncations reset episodes but retain continuation targets.
- Critic output preserves leading time/batch dimensions during imagination.
- Positional encodings grow beyond the original 30-agent limit.
- Bounded replay allocation, finite-gradient checks, learner checkpoints and evaluation.
- Shared normalized advantages and a target critic synchronized after each learner update.

These are MATE defaults, not the paper's tuned Flatland/StarCraft settings.
Training quality on MATE still requires longer runs and hyperparameter evaluation.

## Train

From the repository root with project dependencies installed (`uv sync`):

```powershell
python train.py --algo mamba --device auto --mamba-episodes 1000
```

A small CPU run that actually performs model and policy updates:

```powershell
python train.py --algo mamba --device cpu --mamba-episodes 2 --mamba-episode-limit 4 --mamba-sequence-length 2 --mamba-batch-size 1 --mamba-warmup 0 --mamba-hidden 32 --mamba-categoricals 2 --mamba-classes 4 --mamba-horizon 3 --mamba-ppo-epochs 2 --mamba-log-every 1 --mamba-save-every 1 --mamba-outdir ckpts/mamba-smoke
```

MAMBA options use the `--mamba-` prefix; common `--device`, `--seed`, and
`--wandb-api` also apply. Other algorithms' generic `--steps`, `--batch-size`,
`--warmup`, learning-rate and checkpoint flags do not configure MAMBA.

| Option | Default | Meaning |
| --- | --- | --- |
| `--mamba-config` | `MATE-4v8-9.yaml` | MATE scenario |
| `--mamba-levels` | 3 | 9 discrete actions per camera |
| `--mamba-episodes` | 1000 | Additional episodes to collect |
| `--mamba-episode-limit` | 64 | Maximum steps per episode |
| `--mamba-sequence-length` | 16 | Transitions per training sequence (at least 2) |
| `--mamba-batch-size` | 8 | Sequences per model update |
| `--mamba-warmup` | 256 | Replay transitions before updates; random actions during warmup |
| `--mamba-replay-capacity` | 100000 | Maximum stored transitions |
| `--mamba-updates-per-episode` | 1 | World-model plus imagination updates per episode |
| `--mamba-horizon` | 15 | Imagined states, including the bootstrap state |
| `--mamba-ppo-epochs` | 5 | PPO passes on each imagination batch |
| `--mamba-hidden` | 256 | Network width, divisible by 8 |
| `--mamba-categoricals` / `--mamba-classes` | 32 / 32 | Discrete latent dimensions |
| `--mamba-model-lr` | 0.0002 | World-model learning rate |
| `--mamba-actor-lr` / `--mamba-critic-lr` | 0.0005 / 0.0005 | Actor/critic learning rates |
| `--mamba-log-every` / `--mamba-save-every` | 10 / 100 | Intervals in episodes |
| `--mamba-outdir` | `ckpts/mamba` | Output directory; final checkpoint is `latest.pt` |

`python train.py --help` lists the remaining discount, entropy and gradient options.
Sequences shorter than the requested length are retained but cannot be sampled.

## Checkpoints and evaluation

Resume a run trained with the default architecture:

```powershell
python train.py --algo mamba --mamba-resume ckpts/mamba/latest.pt --mamba-episodes 100
```

When resuming a custom architecture/scenario, pass the same architecture/scenario
options. Checkpoints restore networks, optimizers, counters and entropy; replay
and random-generator states are not saved, so replay warms up again and resumption
is not bitwise identical. Optimizer learning rates come from the checkpoint.
API credentials are excluded from checkpoint data.

Evaluate with the saved configuration automatically restored:

```powershell
python -m mate.algorithms.mamba --checkpoint ckpts/mamba/latest.pt --episodes 20 --device cpu
```

Evaluation reports mean reward, episode length and coverage, uses greedy action
logits and retains stochastic latent sampling as in the upstream world model.
The `--seed` option is an episode-seed offset added to the saved training seed.

For programmatic use: `MAMBAConfig`, `MAMBATrainer`, `trainer.act()`,
`trainer.save()`, and `MAMBATrainer.from_checkpoint()`. Pass returned previous
action and latent state into the next `act()` call; reset both to `None` at each
episode boundary. Close `trainer.env` after programmatic evaluation.

## Verification

```powershell
python -m unittest discover -s tests -p test_mamba.py -v
```

Tests cover temporal alignment, terminal/truncation semantics, replay boundaries
and capacity, analytic lambda returns, critic dimensions, actual MATE transitions,
finite updates to all three networks, checkpoint equality and resumed updates.
