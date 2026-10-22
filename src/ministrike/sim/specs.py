"""Local: versioned game interfaces; global: dimensions/rules follow the selected match.

These are engineering choices from the approved Stage A plan, documented in
docs/duel-protocol.md. Dataclasses: https://docs.python.org/3.12/library/dataclasses.html
"""

from dataclasses import dataclass, replace  # Local: immutable specifications and explicit rule overrides.

import numpy as np  # Local: canonical map arrays are shared by rendering and physics.

from .constants import ACTION_NVECS, HISTORY_LENGTH, GameConfig  # Local: preserve legacy interfaces verbatim.
from .map import SITES, SPAWNS, WALLS  # Local: keep original tactical geometry unchanged.


@dataclass(frozen=True)
class MatchSpec:
    """Local: immutable interface identity; global: checkpoints cannot silently change games."""

    identity: str = "legacy-5v5-v1"  # Local: absent metadata in old checkpoints maps to this version.
    team_size: int = 5  # Local: two equal teams, with stable slots throughout a round.
    bomb: bool = True  # Local: only bomb matches execute the objective state machine.
    map_id: str = "tactical-v1"  # Local: a geometry identifier independent of policy weights.
    observation_id: str = "local-history-v1"  # Local: prefix fields followed by six fields per player.
    action_id: str = "categorical-v1"  # Local: physical decoder version; disabled heads are omitted.

    @property
    def players(self):
        """Local: count permanent slots; global: no fixed ten-player assumptions in shared phases."""
        return self.team_size * 2  # Local: future FFA needs a distinct registered team-assignment interface.

    @property
    def frame_dim(self):
        """Local: size public/personal prefix and entity rows; global: actors use the exact schema."""
        return 36 + 6 * self.players  # Local: historical 5v5 remains 96 features per frame.

    @property
    def observation_dim(self):
        """Local: stack four frames; global: retain the approved limited-memory representation."""
        return HISTORY_LENGTH * self.frame_dim  # Local: 192 duel or 384 legacy features.

    @property
    def action_nvecs(self):
        """Local: remove interaction in deathmatch; global: disabled controls contribute no entropy."""
        return ACTION_NVECS if self.bomb else ACTION_NVECS[:-1]  # Local: actual output dimension changes.

    def geometry(self):
        """Local: provide original map arrays; global: one geometry source for all execution modes."""
        if self.bomb:
            return WALLS, SITES, SPAWNS  # Local: legacy geometry and spawns are unmodified.
        walls = np.array([[-7.5, 0, -7.5, -7, 3.2, 7.5], [7, 0, -7.5, 7.5, 3.2, 7.5],
                          [-7, 0, -7.5, 7, 3.2, -7], [-7, 0, 7, 7, 3.2, 7.5]], np.float32)
        return walls, np.empty((0, 3), np.float32), np.array([[0, 0, -3], [0, 0, 3]], np.float32)

    def rules(self):
        """Local: preserve rifle rules; global: only the explicitly registered round duration changes."""
        return GameConfig() if self.bomb else replace(GameConfig(), round_seconds=20)  # Local: 400 ticks.


def match_spec(identity=None):
    """Local: reject unknown interfaces; global: deferred roadmap entries are not executable games."""
    if identity in (None, "legacy-5v5-v1"):
        return MatchSpec()  # Local: backwards-compatible defaults for old files and callers.
    if identity == "duel-fixed-v1":
        return MatchSpec(identity, 1, False, "flat14-fixed6-v1")  # Local: approved Stage A only.
    raise ValueError(f"Unknown or unimplemented match: {identity}")  # Local: no implicit fallback.
