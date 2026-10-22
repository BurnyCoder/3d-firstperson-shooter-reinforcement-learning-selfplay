"""Local: frozen legal duel evaluation controls; global: these never enter Stage A training.

Original engineering benchmark, registered in docs/duel-protocol.md. NumPy argmin:
https://numpy.org/doc/stable/reference/generated/numpy.argmin.html . All controls
pass the actor's categorical decoder; no finer scripted aiming is possible.
"""

import numpy as np  # Local: deterministic, vectorized observation-only decisions.

from ministrike.sim.constants import decode_actions  # Local: exact shared categorical-to-physical mapping.
from ministrike.sim.specs import match_spec  # Local: baseline dimensions match the frozen game.


class DuelRandomPolicy:
    """Local: seeded categorical actions; global: reproducible untrained behavioral baseline."""

    def __init__(self, seed):
        """Local: dedicate one stream per paired world; global: evaluation batching cannot change trajectories."""
        self.rng = np.random.default_rng(seed)  # Local: never share with simulator or learner randomness.
        self.spec = match_spec("duel-fixed-v1")  # Local: no interaction category.

    def act(self, observation):
        """Local: uniformly sample each legal head; global: no private simulator fields are accessed."""
        actions = self.rng.integers(self.spec.action_nvecs, size=(*observation.shape[:-1], 5))
        return decode_actions(actions, self.spec.action_nvecs)  # Local: exact learner limits and quantization.


class DuelScriptedPolicy:
    """Local: move, acquire visible targets and fire; global: fixed evaluation challenge, no training data."""

    spec = match_spec("duel-fixed-v1")  # Local: class is specific to the registered flat duel, not navigation AI.

    def act(self, observation):
        """Local: use own pose/rays and visible relative positions; global: no hidden state or aiming advantage."""
        frame = np.asarray(observation)[..., -self.spec.frame_dim:]  # Local: same current actor frame.
        targets = frame[..., 36:].reshape(*frame.shape[:-1], 2, 6)  # Local: public team IDs plus masked measurements.
        seen = (targets[..., 3] > .5) & (targets[..., 4] < .5)  # Local: choose only a visible enemy.
        enemy = (targets[..., :3] * seen[..., None]).sum(-2) * 30  # Local: at most one opponent.
        visible = seen.any(-1)  # Local: missing target invokes search, not privileged tracking.
        yaw = np.arctan2(frame[..., 3], frame[..., 4])  # Local: decode own publicly observed orientation.
        pitch = np.arctan2(frame[..., 5], frame[..., 6])  # Local: own pitch is legal input.
        yaw_error = (np.arctan2(enemy[..., 0], enemy[..., 2]) - yaw + np.pi) % (2 * np.pi) - np.pi
        pitch_error = np.arctan2(-.6, np.maximum(np.linalg.norm(enemy[..., [0, 2]], axis=-1), .01)) - pitch
        actions = np.zeros((*frame.shape[:-1], 5), dtype=np.int64)  # Local: default no fire, movement supplied below.
        yaw_steps, pitch_steps = np.array([-.24, -.08, 0, .08, .24]), np.array([-.12, -.04, 0, .04, .12])
        actions[..., 1] = np.abs(yaw_error[..., None] - yaw_steps).argmin(-1)  # Local: nearest legal yaw increment.
        actions[..., 2] = np.abs(pitch_error[..., None] - pitch_steps).argmin(-1)  # Local: nearest legal pitch increment.
        actions[..., 1] = np.where(visible, actions[..., 1], 4)  # Local: turn to search only when sight is lost.
        actions[..., 2] = np.where(visible, actions[..., 2], np.abs(-pitch[..., None] - pitch_steps).argmin(-1))
        phase = np.floor((1 - frame[..., 14]) * 20).astype(int) % 2  # Local: deterministic one-second strafing cycle.
        actions[..., 0] = np.where(phase == 0, 3, 4)  # Local: moving target; period is frozen before learning.
        actions[..., 0] = np.where(frame[..., 28] < .03, 2, actions[..., 0])  # Local: short public ray avoids front walls.
        yaw_residual = yaw_error - yaw_steps[actions[..., 1]]  # Local: shoot based on post-action quantized aim.
        pitch_residual = pitch_error - pitch_steps[actions[..., 2]]
        actions[..., 3] = visible & (np.abs(yaw_residual) < .12) & (np.abs(pitch_residual) < .12)
        actions[..., 4] = (frame[..., 8] < .15) & (frame[..., 9] > 0)  # Local: finite-ammo reload decision.
        return decode_actions(actions, self.spec.action_nvecs)  # Local: no direct continuous scripted controls.
