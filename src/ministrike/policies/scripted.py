"""Local: explicit random/scripted controls. Global: honest benchmarks and pre-training playable bots.

Sources: https://numpy.org/doc/stable/reference/random/generator.html and
https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html .
These controllers access observations only; trained actors never call their aiming or pathfinding.
"""

import numpy as np  # Compute independent player controls in a single batched operation.

from ministrike.policies.navigation import next_waypoint  # Public-map routing belongs only to the scripted baseline.
from ministrike.sim.constants import ACTION_NVECS, FRAME_DIM, decode_actions  # Reuse the exact learned control limits.


class RandomPolicy:
    """Local: seeded random action source. Global: a deliberately weak, reproducible control."""

    def __init__(self, seed: int = 0):
        """Own the RNG so evaluation does not disturb simulation or training seeds."""
        self.rng = np.random.default_rng(seed)  # Never use NumPy's global generator.

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample each discrete control component, then use the shared physical decoder."""
        actions = np.stack([self.rng.integers(size, size=observation.shape[:-1]) for size in ACTION_NVECS], axis=-1)  # Independent categorical heads.
        return decode_actions(actions)  # A random bot has exactly the learned bot's action limits.


class ScriptedPolicy:
    """Local: objective-seeking visible-target baseline. Global: separate fixed-opponent experiment."""

    def __init__(self, style: str = "balanced"):
        """Select a frozen benchmark style without changing observation permissions."""
        if style not in {"balanced", "rush", "guard"}:  # Fail on accidental benchmark-name typos.
            raise ValueError(f"Unknown scripted style: {style}")  # Prevent silently changing the test opponent.
        self.style = style  # All decisions remain deterministic given this fixed style and observations.

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Convert only visible state to raw controls; no private Simulation reference exists."""
        leading = observation.shape[:-1]  # Accept arbitrary environment/player batch dimensions.
        frame = observation[..., -FRAME_DIM:].reshape(-1, FRAME_DIM)  # Current state is the final stacked frame.
        controls = np.zeros((len(frame), 7), dtype=np.float32)  # Dead players return no-op controls.
        if not np.any(frame[:, 11] > 0.5):  # Avoid navigation work on wholly dead batches.
            return controls.reshape(*leading, 7)  # Maintain the caller's original leading dimensions.
        position = frame[:, :3] * 30  # Undo the observation's documented spatial normalization.
        yaw = np.arctan2(frame[:, 3], frame[:, 4])  # Reconstruct heading without angle discontinuity in observations.
        pitch = np.arctan2(frame[:, 5], frame[:, 6])  # Reconstruct vertical aim.
        team = frame[:, 12] > 0.5  # Team identity is public information.
        players = frame[:, 36:96].reshape(-1, 10, 6)  # Fixed slots prevent variable observation lengths.
        enemies = (players[..., 3] > 0.5) & (players[..., 4] < 0.5)  # Hidden enemies are never selected.
        enemy_distance = np.linalg.norm(players[..., :3] * 30, axis=-1)  # Only visible distances can win the masked minimum.
        target_index = np.argmin(np.where(enemies, enemy_distance, np.inf), axis=1)  # Select the nearest observable enemy.
        sees_enemy = np.any(enemies, axis=1)  # Distinguish a real target from argmin's empty default.
        relative_enemy = players[np.arange(len(frame)), target_index, :3] * 30  # Batched gather preserves player identity.
        sites = frame[:, 22:28].reshape(-1, 2, 3) * 30 + position[:, None]  # Both bomb sites are known map information.
        site_index = (position[:, 0] >= 0).astype(int)  # Split the team across the two map lanes.
        target = sites[np.arange(len(frame)), site_index]  # Initial goal is the chosen public site.
        knows_bomb = frame[:, 21] > 0.5  # Never infer a hidden enemy carrier's location.
        planted = frame[:, 19] > 0.5  # A plant announcement is public.
        dropped_for_attack = (~team) & (frame[:, 13] < 0.5) & knows_bomb & (~planted)  # Supporting attackers can recover a visible dropped bomb.
        bomb_target = knows_bomb & (planted | dropped_for_attack)  # Defenders chase only the announced planted device.
        target[bomb_target] = position[bomb_target] + frame[bomb_target, 16:19] * 30  # Recover the known world position.
        waypoint = next_waypoint(position, target)  # Route using public geometry, never enemy state.
        navigation = waypoint - position  # Movement follows the next unobstructed grid cell.
        target_distance = np.linalg.norm((target - position)[:, [0, 2]], axis=-1)  # Stop close enough to interact.
        direction = navigation[:, [0, 2]]  # Ground movement has no vertical component.
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)  # Normalize diagonal movement safely.
        controls[:, 0] = direction[:, 0] * np.sin(yaw) + direction[:, 1] * np.cos(yaw)  # Project world velocity onto local forward.
        controls[:, 1] = direction[:, 0] * np.cos(yaw) - direction[:, 1] * np.sin(yaw)  # Project onto local right.
        near_objective = target_distance < 0.9  # Interaction range is checked authoritatively by the simulator.
        controls[near_objective, :2] = 0  # Stay still while holding a plant or defuse action.
        desired_yaw = np.arctan2(navigation[:, 0], navigation[:, 2])  # Look along the navigation path by default.
        desired_yaw[sees_enemy] = np.arctan2(relative_enemy[sees_enemy, 0], relative_enemy[sees_enemy, 2])  # Scripted aim is explicitly the control baseline.
        desired_pitch = np.zeros(len(frame))  # Level aim supports normal corridor navigation.
        desired_pitch[sees_enemy] = np.arctan2(-0.35, np.maximum(np.linalg.norm(relative_enemy[sees_enemy][:, [0, 2]], axis=-1), 0.1))  # Aim at body height below eye level.
        yaw_error = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi  # Choose the shortest angular turn.
        pitch_error = desired_pitch - pitch  # Pitch does not wrap across the horizon.
        controls[:, 2] = np.clip(yaw_error, -0.24, 0.24)  # Match the trained policy's maximum yaw rate.
        controls[:, 3] = np.clip(pitch_error, -0.12, 0.12)  # Match the trained policy's maximum pitch rate.
        controls[:, 4] = sees_enemy & (np.abs(yaw_error) < 0.18) & (np.abs(pitch_error) < 0.15)  # Fire only after rough alignment.
        controls[:, 5] = (frame[:, 8] < 0.15) & (frame[:, 9] > 0)  # Reload before exhausting the magazine.
        controls[:, 6] = near_objective  # The simulator enforces role, possession, radius, and uninterrupted progress.
        if self.style != "rush":  # Balanced/guard opponents stabilize aim when they spot a target.
            controls[sees_enemy, :2] *= 0.2  # This is a fixed scripted behavior, not hidden assistance to learned bots.
        if self.style == "guard":  # A distinct evaluation opponent waits near its assigned site.
            controls[team & (target_distance < 3) & (~planted), :2] = 0  # Guarding applies only before a plant announcement.
        controls[frame[:, 11] <= 0.5] = 0  # Dead players cannot act, irrespective of stale history.
        return controls.reshape(*leading, 7)  # Restore the environment/player dimensions expected by Simulation.
