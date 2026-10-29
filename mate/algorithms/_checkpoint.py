"""Generic checkpoint save/resume for MATE trainers.

Every PyMARL-style trainer in ``mate.algorithms`` saves its own weights at the
end of ``train()`` and — historically — never loads anything.  This mixin gives
any trainer a consistent, W&B-aware way to:

* periodically save "latest.pt" (and upload it to W&B, ``policy="now"``),
* resume from a previous checkpoint on the next run.

To use it, a trainer must:

* inherit ``CheckpointMixin``,
* implement ``checkpoint_state()``  -> a picklable dict (weights + counters +
  config) describing *its* network and progress,
* implement ``restore_checkpoint(state)`` -> load those weights and reset the
  counters,
* call ``self.save_checkpoint()`` at the end of ``train()`` and
  ``self.maybe_save(counter, cfg.save_every)`` at the periodic point.

The atomic ``.tmp``-then-``replace`` write and the W&B upload are handled here
so every algorithm behaves the same.  A ``wandb`` attribute (a ``WandbLogger``)
is expected; if it is inactive, upload is skipped silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


class CheckpointMixin:
    """Periodic save + resume with W&B upload, shared by all MATE trainers."""

    def _ckpt_path(self, path: Optional[str] = None) -> Path:
        return Path(path) if path else Path(self.config.outdir) / "latest.pt"

    def save_checkpoint(self, path: Optional[str] = None) -> Path:
        """Write ``checkpoint_state()`` to disk atomically and upload to W&B."""
        out = self._ckpt_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        torch.save(self.checkpoint_state(), tmp)
        tmp.replace(out)
        if self.wandb.active:
            try:
                self.wandb.save(out)
            except Exception as exc:  # never block training on W&B failures
                print(f"[wandb] checkpoint upload failed: {exc}", flush=True)
        return out

    def maybe_save(self, counter: int, every: int, path: Optional[str] = None) -> None:
        """Save when ``counter`` is a non-zero multiple of ``every`` (if set)."""
        if every and counter % every == 0:
            self.save_checkpoint(path=path)

    def resume_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load a checkpoint and feed it to ``restore_checkpoint``.

        Returns ``True`` if a checkpoint was applied.  ``path`` defaults to
        ``self.config.resume``; returns ``False`` silently when unset.
        """
        ckpt_path = Path(path) if path else self.config.resume
        if not ckpt_path:
            return False
        state = torch.load(ckpt_path, map_location=self.device)
        self.restore_checkpoint(state)
        print(f"[resume] loaded {ckpt_path}", flush=True)
        return True
