"""FACMAC - Factored Multi-Agent Centralised Policy Gradients - for MATE."""

from mate.algorithms.facmac.facmac import (
    FACMACConfig,
    FACMACDiscreteCritic,
    FACMACTrainer,
    GumbelActor,
    run_facmac,
)

__all__ = [
    "FACMACConfig",
    "FACMACDiscreteCritic",
    "FACMACTrainer",
    "GumbelActor",
    "run_facmac",
]
