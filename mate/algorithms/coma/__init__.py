"""Counterfactual multi-agent policy gradients for MATE."""

from mate.algorithms.coma.coma import COMAConfig, COMATrainer, run_coma

__all__ = ["COMAConfig", "COMATrainer", "run_coma"]
