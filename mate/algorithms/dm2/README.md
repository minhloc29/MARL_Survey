# DM² for MATE

A compact, MATE-native port of **DM² — Distributed Multi-Agent Reinforcement
Learning for Distribution Matching** (Wang et al., [arXiv:2206.00233](https://arxiv.org/abs/2206.00233)),
built on the shared camera-team environment used by the other MATE algorithms
(`make_mate_rode_env`).

The idea: train an on-policy IPPO/MAPPO-style policy while a per-agent GAIL
discriminator reshapes the reward so the policy's state / (state, action)
distribution matches expert demonstrations:

    r_hat = log D - log(1 - D)
    r     = env_r                          (rew_type="env")
          = r_hat                          (rew_type="gail")
          = r_hat * gail_rew_coef + env_r  (rew_type="mixed")   <-- headline

The discriminator uses binary cross-entropy plus a WGAN gradient penalty. It
also supports **SIL** (self-imitation learning, no external demos) via the
return-ranked `SILStorage`.

Only `numpy` + `torch` are required.

## Usage

```python
from mate.algorithms.dm2 import DM2Config, DM2Trainer

cfg = DM2Config(
    rew_type="mixed",                    # "env" | "gail" | "mixed"
    gail_data_paths=["/path/to/agent_0"],  # expert demos (npz/npy per agent)
    gail_exp_eps=1000,
    episodes=1000,
    device="cpu",
)
DM2Trainer(cfg).train()
```

- **With demonstrations** (main method): set `gail_data_paths` to your expert
  demo dirs (`.npz`/`.npy`), `gail_exp_eps` demonstrations per agent, and
  `rew_type="mixed"` (or `"gail"`).
- **SIL** (no demos): set `gail_sil=True`; the discriminator matches the
  agent's own best past episodes.
- **Plain IPPO** baseline: set `rew_type="env"` (GAIL is off).

Checkpoints (actor/critic/discriminators) are written to `cfg.outdir`.
