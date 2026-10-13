"""Optional W&B integration shared by the MATE algorithm trainers.

Every trainer in ``mate.algorithms`` prints its metrics to stdout. This small
wrapper additionally pushes those same metrics to W&B when an API key is
provided, and is a strict no-op otherwise — training never breaks because of a
W&B failure.
"""

from __future__ import annotations

from typing import Any, Optional


class WandbLogger:
    """Lazy wrapper around a ``wandb.Run`` that no-ops without an API key."""

    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._run: Any = None
        if not api_key:
            return
        try:
            import wandb

            wandb.login(key=api_key, relogin=True)
            self._run = wandb.init(
                project=project, name=name, config=config or {}, reinit=True)
        except Exception as exc:  # never block training on W&B issues
            print(f"[wandb] disabled for {name}: {exc}", flush=True)
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is not None:
            self._run.log(data, step=step)

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # pragma: no cover - best-effort
                pass
