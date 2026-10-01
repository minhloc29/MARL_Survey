# MAMBPO for MATE

This MATE-native implementation follows
[danielwillemsen/MAMBPO](https://github.com/danielwillemsen/MAMBPO), commit
`61593227dfb8ee6cfc21c0e5bb81098f417a6a5a`. The original checkout is at
`D:\mambpo`. This package runs independently of that checkout and its legacy
particle-environment submodules.

## Implementation

MAMBPO uses one probabilistic world-model ensemble shared by the camera team,
separate decentralized actors, and separate centralized twin critics per camera.
The integration uses the repository's continuous SAC branch: each actor maps its
own rescaled camera observation to a tanh-squashed Gaussian action. The two action
components control camera orientation and viewing angle. Normalized actions in
`[-1, 1]` are mapped to each camera's actual action-space bounds by
`ContinuousCameraEnvironment`.

| Upstream source | Integrated component |
| --- | --- |
| `decentralizedlearning/algs/utils.py::StochActor` | `networks.py::GaussianActor` |
| `algs/mambpo.py::MAMBPOAgent`, `initialize_agents` | `networks.py::MAMBPOAgent`, independent actor/critic targets |
| `algs/models.py::Model`, `EnsembleModel.forward` | `networks.py::ResidualModel`, `WorldModel` |
| `algs/models.py::generate_efficient` | `mambpo.py::rollout_model` |
| `algs/mambpo.py::update_actor`, `update_critics` | `mambpo.py::update_actors`, `update_critics` |
| `algs/mambpo.py::step`, `sample_batch` | Step-based learning schedules and shared real/synthetic replay |

The shared world model predicts the **change in joint observations** and a scalar
team reward. It consists of independently initialized MLPs with bounded Gaussian
log variances; all configured members are eligible for sampling, matching the
active upstream training path. Model training uses Gaussian negative log
likelihood. A member is sampled independently for each synthetic trajectory.

Synthetic trajectories begin from real replay observations. Every camera selects
its own action from its own simulated observation. The default rollout length is
one step, as in the upstream thesis experiments. Real and generated transitions
are mixed for SAC updates. Critic targets use all target actors and the ego
camera's entropy term. Actor updates replace only that camera's action in the
replayed joint action, holding the other cameras' actions fixed. Actor, target
actor and target critic updates follow the configured delay.

Execution is decentralized at the actor input: changing another camera's input
does not change this camera's greedy action. Joint observations/actions are used
by the world model and training critics. Targets use `GreedyTargetAgent`; the
camera-team reward is shared by all learning cameras.

## MATE adaptations

- Python 3.13, modern PyTorch and Gymnasium, using existing project dependencies.
- Continuous camera control; the earlier MAMBA/AORPO integrations retain their
  discrete camera grids. The action spaces differ when comparing these runs.
- Real transitions are saved immediately after `env.step`, including the final
  episode transition. Every fifth transition is reserved for model validation;
  SAC may still sample all real transitions.
- A learned terminal head is added for MATE. Predicted terminal states stop model
  rollout and suppress TD bootstrapping. Runner time limits permit bootstrapping.
- Configurable ensemble size without the upstream fixed-ten-member indexing.
  Configurable rollout length is passed through and **all** generated steps are
  retained, including multi-step rollouts.
- Shared lazy replay in `mate/algorithms/model_based_replay.py` preserves floating
  actions. AORPO's existing replay import is retained as a compatibility alias.
- Finite-gradient checks, numerical tanh Jacobian correction, registered target
  networks/temperature parameters and complete learner checkpoints.
- Bounded model training steps, model-refit intervals and actor/critic schedules
  are configurable. These are MATE defaults, not the thesis's tuned settings.

The core path is the independent-actor, independent-central-critic configuration.
The upstream optional shared-policy/shared-critic experiments and discrete actor
branch are not exposed here. The original particle/MuJoCo checkpoints have
different observation/action spaces and cannot initialize this MATE controller.

## Train

From the repository root, with the project environment activated:

```powershell
python train.py --algo mambpo --device auto --mambpo-episodes 1000
```

In the local environment prepared during integration, replace `python` with
`.\.venv\Scripts\python.exe`.

CPU smoke run with actual model, critic and actor updates:

```powershell
python train.py --algo mambpo --device cpu --mambpo-episodes 2 --mambpo-episode-limit 8 --mambpo-actor-hidden 32 --mambpo-critic-hidden 32 --mambpo-model-hidden 32 --mambpo-ensemble-size 2 --mambpo-batch-size 4 --mambpo-model-batch-size 4 --mambpo-model-steps 1 --mambpo-policy-updates 2 --mambpo-warmup 4 --mambpo-rollout-batch-size 4 --mambpo-update-every 4 --mambpo-log-every 1 --mambpo-save-every 1 --mambpo-outdir ckpts/mambpo-smoke
```

All algorithm-specific flags use `--mambpo-`. Common `--device`, `--seed` and
`--wandb-api` apply. Other algorithms' generic `--steps`, `--batch-size`,
`--warmup`, learning-rate and checkpoint flags do not configure MAMBPO.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mambpo-config` | `MATE-4v8-9.yaml` | MATE scenario |
| `--mambpo-episodes` | 1000 | Additional episodes to collect |
| `--mambpo-episode-limit` | 64 | Maximum steps per episode |
| `--mambpo-actor-hidden` / `--mambpo-critic-hidden` | 128 / 256 | Actor/critic widths, two hidden layers |
| `--mambpo-model-hidden` | 200 | Model width, four hidden layers |
| `--mambpo-ensemble-size` | 10 | Shared dynamics ensemble members |
| `--mambpo-warmup` | 500 | Real transitions collected before learning; random actions during warmup |
| `--mambpo-batch-size` | 64 | Mixed transitions per SAC gradient step |
| `--mambpo-real-ratio` | 0.1 | Real fraction, floored to a transition count |
| `--mambpo-model-batch-size` | 128 | Real training records per model gradient step |
| `--mambpo-model-steps` | 20 | Gradient steps per model refit |
| `--mambpo-model-update-every` | 250 | Real steps between model refits, checked at learning cycles |
| `--mambpo-update-every` | 10 | Real steps between learning cycles |
| `--mambpo-policy-updates` | 10 | Critic gradient steps per learning cycle |
| `--mambpo-actor-delay` | 2 | Critic steps per actor/target update |
| `--mambpo-rollout-batch-size` | 256 | Real starting observations per model rollout |
| `--mambpo-rollout-length` | 1 | Maximum synthetic trajectory length |
| `--mambpo-replay-capacity` | 100000 | Capacity of real replay and model-training replay each |
| `--mambpo-model-capacity` | 10000 | Shared synthetic replay capacity |
| `--mambpo-actor-lr` / `--mambpo-critic-lr` | 0.0003 / 0.0003 | Policy/value learning rates |
| `--mambpo-model-lr` | 0.001 | World-model learning rate |
| `--mambpo-alpha` | 0.1 | Initial entropy temperature |
| `--mambpo-autotune` | enabled | Learn temperature; disable with `--no-mambpo-autotune` |
| `--mambpo-target-entropy` | -2 | Entropy target for the two continuous action dimensions |
| `--mambpo-log-every` / `--mambpo-save-every` | 10 / 100 | Intervals in episodes |
| `--mambpo-outdir` | `ckpts/mambpo` | Output directory, final checkpoint `latest.pt` |

Synthetic replay is refreshed each learning cycle. Logging reports the latest
losses generated during an episode and cumulative model/actor/critic update
counts. W&B is enabled through `--wandb-api` or the existing `WANDB_API_KEY`
environment variable. Credentials are excluded from saved configuration.

## Resume and evaluate

Resume a default-architecture run:

```powershell
python train.py --algo mambpo --mambpo-resume ckpts/mambpo/latest.pt --mambpo-episodes 100
```

When resuming a custom run, repeat its scenario, architecture and autotune flags.
Other training settings can be supplied as desired; optimizer learning rates
come from the checkpoint. The checkpoint stores the ensemble, actors, critics,
targets, temperatures, optimizer state and schedule counters. Replay and random
generator states are not saved, so resumed training collects fresh warmup data
and refits the loaded model before synthetic rollouts. It is not a bitwise
continuation.

Evaluate greedy local actors with the saved configuration:

```powershell
python -m mate.algorithms.mambpo --checkpoint ckpts/mambpo/latest.pt --episodes 20 --device cpu
```

Evaluation reports mean team reward, episode length and coverage. It neither
updates networks nor writes replay. The `--seed` option is an offset added to
the saved training seed.

Programmatic entry points: `MAMBPOConfig`, `MAMBPOTrainer`,
`trainer.act(obs, greedy=True)`, `trainer.save(path)` and
`MAMBPOTrainer.from_checkpoint(path)`. An individual camera can directly call
`trainer.agents[i].actor(own_observation_batch, greedy=True)`. Close `trainer.env`
after programmatic evaluation.

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The 12 MAMBPO tests cover actor locality, normalized continuous actions,
fractional replay data, stable squashed-Gaussian log probabilities, residual
state prediction, shared rewards, multi-step rollout retention, terminal handling,
ego entropy in critic targets, delayed target updates, real MATE learning,
checkpoint restoration and single-camera fixed-temperature learning. The full
31-test suite includes the existing AORPO and MAMBA tests.

Smoke tests establish executable integration and numerical behavior. Long-run
policy performance on MATE has not yet been benchmarked.
