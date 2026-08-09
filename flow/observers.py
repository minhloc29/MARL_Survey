"""Parse MATE's raw joint camera observation into per-camera entity dicts.

MATE gives each camera a fixed-length concatenated observation vector
(dimension depends on the config). We rely on ``mate``'s own
``observation_slices_of`` to slice it into its parts, then re-pack the three
variable-dim entity sets (targets / obstacles / teammates) into fixed-maximum
padded matrices plus boolean validity masks, ready for an ``EntitySetEncoder``.
"""

from __future__ import annotations

import numpy as np

from mate.constants import Team, observation_slices_of


class MateObserver:
    """Turn a raw (num_cameras, obs_dim) joint observation into per-camera dicts.

    Args:
        num_cameras: number of camera agents.
        num_targets: number of targets.
        num_obstacles: number of obstacles.
    """

    def __init__(self, num_cameras: int, num_targets: int, num_obstacles: int):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles

        slices = observation_slices_of(
            Team.CAMERA, num_cameras, num_targets, num_obstacles
        )
        self._slices = {
            "self_state": slices["self_state"],
            "targets": slices["opponent_states_with_mask"],
            "obstacles": slices["obstacle_states_with_mask"],
            "teammates": slices["teammate_states_with_mask"],
        }

    @staticmethod
    def _to_entities(flat: np.ndarray, n_entities: int) -> np.ndarray:
        """Reshape a flat entity-with-mask vector into (n_entities, per_entity_dim)."""
        return flat.reshape(n_entities, -1)

    def parse(self, joint_obs: np.ndarray) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Parse a joint camera observation.

        Args:
            joint_obs: (num_cameras, obs_dim) raw camera observation.

        Returns:
            A list of ``num_cameras`` dicts, each keyed by entity kind and
            holding ``(entities, valid_mask)`` tuples. ``valid_mask`` is the
            sensed bit (last column of each entity row) — rows outside a
            camera's sensing range are masked out:
              - "targets":    (num_targets, 5), mask from col -1
              - "obstacles":  (num_obstacles, 4), mask from col -1
              - "teammates":  (num_cameras, 7), mask from col -1 (incl. self)
        """
        results = []
        for c in range(self.num_cameras):
            obs = np.asarray(joint_obs[c], dtype=np.float64)

            self_state = obs[self._slices["self_state"]]  # (9,)

            items = {}
            for kind, n_ent in (
                ("targets", self.num_targets),
                ("obstacles", self.num_obstacles),
                ("teammates", self.num_cameras),  # includes self
            ):
                raw = obs[self._slices[kind]]
                entities = self._to_entities(raw, n_ent)
                # The *_with_mask slices append a sensed/seen bit as the LAST
                # column of each entity row. Rows whose bit is 0 are out of a
                # camera's sensing range (stale/zero-filled) and must be masked
                # out of the entity encoder's attention/pooling.
                valid = entities[:, -1].astype(bool)
                items[kind] = (entities, valid)

            results.append({"self_state": self_state, "entities": items})
        return results
