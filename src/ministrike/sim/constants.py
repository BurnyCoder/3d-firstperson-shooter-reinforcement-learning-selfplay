"""Local: shared rules and tensor layout. Global: rendering and RL consume one game contract.

NumPy broadcasting conventions: https://numpy.org/doc/stable/user/basics.broadcasting.html
"""

from dataclasses import dataclass  # Local: immutable rule configuration. Global: reproducible experiments.

import numpy as np  # Local: vectorized action decoding. Global: headless and human physics share units.

DT = 0.05  # Local: fixed twenty-Hertz tick. Global: rendering cannot change simulation time.
EYE_HEIGHT = 1.5  # Local: camera/ray origin above floor in metres.
PLAYER_HEIGHT = 1.8  # Local: vertical hitbox size in metres.
PLAYER_RADIUS = 0.3  # Local: half-width of player collision/hit boxes.
FRAME_DIM = 96  # Local: one observation frame width; see OBS_LAYOUT for field meanings.
HISTORY_LENGTH = 4  # Local: four fixed-tick frames provide limited motion information.
OBS_DIM = FRAME_DIM * HISTORY_LENGTH  # Local: flattened actor observation width.
ACTION_NVECS = (9, 5, 5, 2, 2, 2)  # Local: move, yaw, pitch, fire, reload, interact cardinalities.
RAY_ANGLES = np.arange(-4, 4, dtype=np.float32) * np.pi / 4  # Local: eight evenly spaced map probes.
OBS_LAYOUT = {  # Local: executable documentation for policies. Global: avoid duplicated index guesses.
    "position": slice(0, 3), "sin_yaw": 3, "cos_yaw": 4, "sin_pitch": 5, "cos_pitch": 6,
    "health": 7, "ammo": 8, "reserve": 9, "reload": 10, "alive": 11, "team": 12,
    "carrying": 13, "round_remaining": 14, "interaction": 15, "bomb_relative": slice(16, 19),
    "planted": 19, "bomb_remaining": 20, "bomb_known": 21, "sites": slice(22, 28),
    "rays": slice(28, 36), "players": slice(36, 96),
}  # Local: player rows are [relative XYZ/30, visible, same_team, visible teammate health/100].


@dataclass(frozen=True)
class GameConfig:
    """Local: configurable timing and weapons. Global: persist exact rules with every checkpoint."""

    health: float = 100.0  # Local: initial hit points for all ten actors.
    damage: float = 25.0  # Local: per-hit rifle damage, without random spread.
    shots_per_second: float = 6.0  # Local: minimum interval between successful shots.
    magazine: int = 30  # Local: maximum ammunition held in the rifle.
    reserve_ammo: int = 90  # Local: finite spare ammunition available each round.
    reload_seconds: float = 2.0  # Local: uninterrupted reload completion time.
    movement_speed: float = 5.0  # Local: floor movement in metres per second.
    round_seconds: float = 60.0  # Local: pre-plant deadline; planted bombs ignore this deadline.
    bomb_seconds: float = 25.0  # Local: armed bomb countdown until attack victory.
    plant_seconds: float = 2.0  # Local: held stationary interaction required to plant.
    defuse_seconds: float = 3.0  # Local: held stationary interaction required to defuse.
    interaction_radius: float = 1.5  # Local: legal planting/defusing distance in metres.
    fov_degrees: float = 110.0  # Local: full three-dimensional cone of player visibility.


def decode_actions(actions):
    """Local: convert categorical heads to physical controls. Global: one decoder serves all learned policies."""
    actions = np.asarray(actions, dtype=np.int64)  # Local: indexing requires integer action categories.
    if actions.shape[-1] != 6 or np.any(actions < 0) or np.any(actions >= ACTION_NVECS):
        raise ValueError("Expected six categorical actions within ACTION_NVECS")  # Local: reject invalid policy outputs.
    moves = np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 1], [-1, -1], [-1, 1]], dtype=np.float32)
    result = np.zeros((*actions.shape[:-1], 7), dtype=np.float32)  # Local: forward/strafe/angles/buttons format.
    result[..., :2] = moves[actions[..., 0]]  # Local: diagonal speed is normalized by movement physics.
    result[..., 2] = np.array([-.24, -.08, 0, .08, .24], dtype=np.float32)[actions[..., 1]]  # Local: radians per tick.
    result[..., 3] = np.array([-.12, -.04, 0, .04, .12], dtype=np.float32)[actions[..., 2]]  # Local: vertical radians per tick.
    result[..., 4:] = actions[..., 3:]  # Local: binary fire/reload/interact become continuous control switches.
    return result  # Local: caller submits this exact array to Simulation.step.
