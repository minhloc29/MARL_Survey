"""DM² - Distributed Multi-Agent RL for Distribution Matching - for MATE."""

from mate.algorithms.dm2.dm2 import (
    DM2Actor,
    DM2Config,
    DM2Critic,
    DM2Discriminator,
    DM2Storage,
    DM2Trainer,
    SILStorage,
    run_dm2,
)

__all__ = [
    "DM2Actor",
    "DM2Config",
    "DM2Critic",
    "DM2Discriminator",
    "DM2Storage",
    "DM2Trainer",
    "SILStorage",
    "run_dm2",
]
