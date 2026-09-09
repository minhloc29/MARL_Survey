# RODE integration

This package is the MATE-native boundary for porting RODE. It does not import
RODE's legacy PyMARL runtime or its obsolete dependency set.

## Current integration

`make_mate_rode_env()` creates a camera-team task with discrete actions and
exposes the environment methods expected by RODE:

```python
from mate.algorithms.rode import make_mate_rode_env

env = make_mate_rode_env(config="MATE-1v2-0.yaml", levels=3)
obs, state = env.reset(seed=0)
reward, done, info = env.step([0] * env.n_agents)
```

The adapter provides:

- fixed homogeneous per-camera observations;
- a flattened centralized state;
- discrete action indices converted to MATE camera vectors;
- dense action-availability masks;
- fixed episode-limit handling;
- Gymnasium reset/step behavior internally, with RODE's three-value step result
  at the adapter boundary.

`RODEEpisodeBuilder` and `RODEEpisodeReplay` preserve complete variable-length
episodes and pad sampled sequences for recurrent role/value learners.

## Not yet enabled

The role-conditioned controller and RODE learner still need to be ported to
modern PyTorch before adding `--algo rode` to `train.py`. The existing MATE
QMIX/QPLEX implementations remain the comparison baselines during that port.