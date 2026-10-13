# FACMAC for MATE

A compact, MATE-native port of **FACMAC — Factored Multi-Agent Centralised
Policy Gradients** (Peng et al., NeurIPS 2021, [arXiv:2003.06709](https://arxiv.org/abs/2003.06709)),
built on the shared camera-team environment used by the other MATE algorithms
(`make_mate_rode_env`).

FACMAC is an off-policy multi-agent policy-gradient method (MADDPG-style)
with two distinguishing pieces:

- a **factorized central critic** — one per-agent Q-value that conditions on
  the *joint* observation and *joint* action (centralised but factored);
- a **gumbel-softmax reparameterisation** of the discrete policies, so the
  actor's one-hot actions are differentiable functions of the policy logits
  and the policy gradient flows through the critic:

```
actor_loss   = -Q(joint_obs, gumbel(pi))
critic_loss  = MSE( Q(joint_obs, a) - TD(lambda)-target )
```

Only `numpy` + `torch` are required.

## Usage

```python
from mate.algorithms.facmac import FACMACConfig, FACMACTrainer

cfg = FACMACConfig(
    config="MATE-4v8-9.yaml",
    levels=3,
    episodes=2000,
    device="cpu",
)
trainer = FACMACTrainer(cfg)
trainer.train()
```

Hyperparameters (gamma, td_lambda, learning rates, target-update cadence,
epsilon schedule, buffer size) mirror the original FACMAC-SMAC config and can
be overridden on the dataclass. Checkpoints (actor/critic) are written to
`cfg.outdir`; call `trainer.evaluate(n)` for a greedy-policy evaluation.
