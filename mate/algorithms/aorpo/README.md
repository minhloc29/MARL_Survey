# AORPO for MATE

This is a MATE-native reimplementation guided by
[apexrl/AORPO](https://github.com/apexrl/AORPO), commit
`6310b678a076cf04385b98474bc710a95a2cbefb`, and
[Model-based Multi-agent Policy Optimization with Adaptive Opponent-wise Rollouts](https://arxiv.org/abs/2105.03363).
The original repository, including its LFS checkpoints, is cloned at `D:\AORPO`.
The integrated package is self-contained and uses the project's existing
NumPy/PyTorch/Gymnasium dependencies.

## Algorithm and source mapping

Each camera has its own conditional actor, twin Q networks and target Q networks,
learned SAC temperature, models of every other camera's policy, probabilistic
ensemble dynamics model, and synthetic replay buffer. Camera teammates take the
role of the upstream algorithm's "opponents". The targets are controlled by MATE's
`GreedyTargetAgent` and their behavior is absorbed into the dynamics model.

| Upstream component | MATE implementation |
| --- | --- |
| `agents/sacagent.py::SACAgentOppMdCond` | `networks.py::AORPOAgent`, `aorpo.py::update_policy` |
| `agents/baseagent.py::AgentMB`, `utils/networks.py::PtModel` | `networks.py::DynamicsEnsemble` |
| `algorithms/maframework.py::evaluate_opp_model` | `aorpo.py::fit_models`, held-out action match rates |
| `agents/baseagent.py::rollout_model` | `aorpo.py::rollout_agent`, adaptive policy queries |
| `main.py` model/real sampling and rollout schedule | `aorpo.py::update`, `horizon`, `replay.py::mixed_batch` |

Training proceeds as follows:

1. Collect rescaled MATE observations, discrete camera actions and team rewards.
   Warmup actions are random. Every fifth transition is reserved for validating
   dynamics and opponent models; all transitions remain available to SAC.
2. Fit each camera's ensemble on bootstrapped real training data and its opponent
   models by categorical behavior cloning with entropy regularization.
3. Measure each opponent model's held-out action prediction accuracy. Following
   the repository's practical approximation, its modeled-action budget is
   `floor(K * accuracy_j / max_accuracy)`. If all accuracies are zero, all budgets
   are zero. This uses the repo's match-rate heuristic rather than claiming to
   estimate the paper's theoretical total-variation error bound.
4. Start model rollouts at real observations. Use an opponent model for exactly
   its budgeted number of steps; thereafter query that camera's current policy
   on the simulated joint observation. These in-process queries do not step MATE.
5. Mix real transitions with that ego camera's synthetic transitions and update
   its conditional SAC actor, twin critics, temperature and target networks.

Like the upstream conditional controller, inference uses the joint observations
to predict other agents' actions. This integration assumes those observations are
available to the controller; it is not an isolated local-observation policy.

## MATE adaptations

- Shared environment boundary in `mate/algorithms/camera_environment.py`; MAMBA
  retains its existing `MAMBAEnvironment` import as a compatibility alias.
- Modern PyTorch truncated-normal initialization and LayerNorm, including
  single-state/single-camera operation; no TensorFlow initialization dependency.
- Absolute next-observation and per-camera reward prediction as in the repo.
  Gaussian sampling uses `exp(logvar / 2)` as the standard deviation.
- Added learned terminal logits for MATE. True terminal transitions stop model
  rollouts and suppress TD bootstrapping; runner time limits permit bootstrapping.
- Straight-through categorical Gumbel actions with categorical log probabilities,
  stochastic next actions for SAC targets, and a configurable positive categorical
  target entropy (`target_entropy_ratio * log(number_of_actions)`).
- Correct zero-based rollout budget boundary and exact per-trajectory policy-query
  counters. Fixed training-epoch limits replace upstream's open-ended early-stop
  loops. Model validation MSE and opponent match rates are logged.
- Synthetic replay is refreshed after each model refit, with a capacity limit per
  ego. The transition buffers allocate storage only as data is collected.
- Complete learner checkpoints, including dynamics, normalization statistics,
  target critics, temperatures, opponent models, query counters and optimizers.

Default widths, training schedules and rollout limits are MATE settings, not the
paper's tuned particle-environment hyperparameters. The integration has been
tested for execution and numerical correctness; policy quality needs longer
training and evaluation. The upstream particle-environment checkpoints have
different observation/action spaces and cannot initialize this MATE controller.

## Train

From the repository root, with the project's dependencies installed and environment
activated:

```powershell
python train.py --algo aorpo --device auto --aorpo-episodes 1000
```

For the local environment prepared during integration, `python` can be replaced
with `.\.venv\Scripts\python.exe`.

Small CPU smoke run with real model and policy updates:

```powershell
python train.py --algo aorpo --device cpu --aorpo-episodes 2 --aorpo-episode-limit 8 --aorpo-hidden 32 --aorpo-model-hidden 32 --aorpo-ensemble-size 2 --aorpo-batch-size 4 --aorpo-model-batch-size 4 --aorpo-model-samples 8 --aorpo-model-epochs 1 --aorpo-opponent-epochs 1 --aorpo-policy-updates 2 --aorpo-warmup 4 --aorpo-rollout-batch-size 4 --aorpo-rollout-min 3 --aorpo-rollout-horizon 3 --aorpo-log-every 1 --aorpo-save-every 1 --aorpo-outdir ckpts/aorpo-smoke
```

All algorithm-specific flags use `--aorpo-`. Common `--device`, `--seed` and
`--wandb-api` apply; other algorithms' generic `--steps`, `--lr`, `--warmup`,
`--batch-size` and `--save-every` do not configure AORPO.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--aorpo-config` | `MATE-4v8-9.yaml` | Scenario |
| `--aorpo-levels` | 3 | 9 discrete camera actions |
| `--aorpo-episodes` | 1000 | Additional episodes |
| `--aorpo-episode-limit` | 64 | Steps per episode |
| `--aorpo-hidden` / `--aorpo-model-hidden` | 64 / 128 | Policy/critic and dynamics widths |
| `--aorpo-ensemble-size` | 5 | Independent dynamics ensemble members per camera |
| `--aorpo-warmup` | 256 | Real transitions before learner updates |
| `--aorpo-batch-size` | 64 | Mixed transitions per SAC update |
| `--aorpo-model-batch-size` | 128 | Training transitions sampled per model step |
| `--aorpo-model-samples` | 512 | Maximum recent training and validation records to consider |
| `--aorpo-model-epochs` / `--aorpo-opponent-epochs` | 5 / 5 | Gradient steps per model refit |
| `--aorpo-policy-updates` | 5 | SAC updates per camera per training episode |
| `--aorpo-real-ratio` | 0.5 | Real fraction; count is floored, remainder is synthetic |
| `--aorpo-replay-capacity` | 100000 | Capacity of real replay and model-training replay each |
| `--aorpo-model-capacity` | 10000 | Synthetic replay capacity per camera |
| `--aorpo-rollout-batch-size` | 64 | Real starting observations per ego rollout |
| `--aorpo-rollout-min` / `--aorpo-rollout-horizon` | 1 / 5 | Initial and maximum dynamics horizon K |
| `--aorpo-rollout-start` / `--aorpo-rollout-end` | 20 / 100 | Episodes over which K grows linearly |
| `--aorpo-lr` | 0.0003 | Actor, critics and temperature learning rate |
| `--aorpo-model-lr` / `--aorpo-opponent-lr` | 0.001 / 0.0005 | Model learning rates |
| `--aorpo-log-every` / `--aorpo-save-every` | 10 / 100 | Intervals in episodes |
| `--aorpo-outdir` | `ckpts/aorpo` | Output directory, final checkpoint `latest.pt` |

Use `python train.py --help` for discount, entropy, reward-scale and target-update
options. W&B is enabled only when `--wandb-api` or the existing `WANDB_API_KEY`
environment variable is provided. Credentials are excluded from checkpoints.

## Resume and evaluate

Resume a default-architecture run:

```powershell
python train.py --algo aorpo --aorpo-resume ckpts/aorpo/latest.pt --aorpo-episodes 100
```

For custom scenarios or architectures, repeat the matching `--aorpo-config`,
`--aorpo-levels`, `--aorpo-hidden`, `--aorpo-model-hidden` and
`--aorpo-ensemble-size` flags. Set other training options as desired; optimizer
learning rates are restored from the checkpoint. Replay and random-generator
states are not saved, so resumed runs collect fresh warmup data and are not
bitwise continuations.

Evaluate using the saved configuration and greedy actions:

```powershell
python -m mate.algorithms.aorpo --checkpoint ckpts/aorpo/latest.pt --episodes 20 --device cpu
```

This prints mean reward, episode length and coverage. `--seed` is an offset added
to the saved training seed. Evaluation does not update networks or replay.

Programmatic use: `AORPOConfig`, `AORPOTrainer`, `trainer.act(obs, greedy=True)`,
`trainer.save(path)` and `AORPOTrainer.from_checkpoint(path)`. Close `trainer.env`
after programmatic evaluation.

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The 10 AORPO tests cover adaptive horizons and exact query counts, Gaussian scale,
singleton normalization, terminal model rollouts and TD targets, replay eviction
and real/model mixtures, categorical gradients, real MATE learning, single-camera
learning, checkpoint equality and resumed temperature updates. The shared
environment also passes the existing 9 MAMBA tests.
