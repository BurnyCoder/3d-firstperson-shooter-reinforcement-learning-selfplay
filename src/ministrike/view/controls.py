"""Local: translate human input into physical controls; global: simulation owns all motion.

Source: https://www.ursinaengine.org/api_reference.html#mouse
"""

from collections.abc import Mapping  # Local: accept Ursina's held-key dictionary or plain test input.
import math  # Local: convert degrees to radians, the simulation's angle convention.

import numpy as np  # Local: return the same float32 action arrays used by learned bots.


class HumanControls:
    """Local: accumulate per-frame mouse deltas; global: consume rotation once per simulation tick."""

    def __init__(self, sensitivity: float = 80.0):
        """Local: configure degrees per normalized cursor unit; global: avoid frame-dependent rotation."""
        self.sensitivity = math.radians(sensitivity)  # Local: store radians per mouse unit.
        self.pending = np.zeros(2, dtype=np.float32)  # Local: keep yaw and pitch awaiting a tick.

    def add_mouse(self, x: float, y: float) -> None:
        """Local: add the latest cursor motion; global: preserve input between fixed-rate updates."""
        self.pending += np.asarray([x, y], dtype=np.float32) * self.sensitivity  # Local: integrate deltas.

    def clear(self) -> None:
        """Local: reset accumulated rotation; global: ignore stale mouse motion when pausing."""
        self.pending.fill(0.0)  # Local: preserve the small allocation while resetting both axes.

    def consume(self, keys: Mapping[str, float]) -> np.ndarray:
        """Local: return forward/strafe/yaw/pitch/fire/reload/interact; global: match the sim contract."""
        controls = np.asarray([  # Local: map held keys into the seven public physical controls.
            keys.get("w", 0) - keys.get("s", 0),  # Local: forward is positive along the current heading.
            keys.get("d", 0) - keys.get("a", 0),  # Local: strafing is positive toward the right.
            self.pending[0],  # Local: yaw is positive toward world +X when facing world +Z.
            self.pending[1],  # Local: pitch is positive upward, matching the simulation.
            keys.get("left mouse", 0),  # Local: held fire requests the simulation's rifle cooldown.
            keys.get("r", 0),  # Local: held reload requests the simulation's ammunition transition.
            keys.get("e", 0),  # Local: held interaction requests planting, pickup, or defusing.
        ], dtype=np.float32)  # Local: keep policy and human outputs interchangeable.
        self.clear()  # Local: subsequent catch-up ticks cannot reuse the same mouse delta.
        return controls  # Local: movement and weapon legality remain simulation decisions.
