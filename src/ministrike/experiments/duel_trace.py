"""Local: read-only action traces for frozen diagnostics; global: privileged audit state never feeds a policy.

Geometry and visibility reuse the authoritative simulator's constants/functions. Angular error is
descriptive only: https://numpy.org/doc/stable/reference/generated/numpy.arccos.html .
"""

import numpy as np  # Local: vector geometry without sampling any RNG.

from ministrike.sim.constants import EYE_HEIGHT, PLAYER_HEIGHT  # Local: exactly the game's eye and body dimensions.
from ministrike.sim.geometry import aim_directions  # Local: shared yaw/pitch convention.


def trace_step(sim, observations, controls, side):
    """Local: copy arena zero before its action; global: traces document behavior without influencing the round."""
    players = []  # Local: include both controllers to expose interaction-dependent loops.
    for player in (0, 1):
        enemy = 1 - player  # Local: this diagnostic is explicitly the two-player match.
        frame = observations[0, player].reshape(-1, sim.spec.frame_dim)[-1]  # Local: the actor's latest legal observation.
        visible = bool(frame[36 + enemy * 6 + 3])  # Local: do not call hidden aim a visible shooting opportunity.
        target = sim.positions[0, enemy] + np.array([0, PLAYER_HEIGHT / 2, 0])
        origin = sim.positions[0, player] + np.array([0, EYE_HEIGHT, 0])
        direction = target - origin  # Local: diagnostic geometry remains separate from action inference.
        cosine = np.dot(direction, aim_directions(sim.yaw[0, player], sim.pitch[0, player])) / max(np.linalg.norm(direction), 1e-8)
        players.append({"position": sim.positions[0, player].tolist(), "yaw": float(sim.yaw[0, player]),
                        "pitch": float(sim.pitch[0, player]), "health": float(sim.health[0, player]),
                        "ammo": int(sim.ammo[0, player]), "reload_seconds": float(sim.reload_timer[0, player]),
                        "cooldown_seconds": float(sim.cooldown[0, player]), "visible": visible,
                        "aim_error_degrees": float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))) if visible else None,
                        "control": controls[0, player].tolist()})  # Local: JSON values are copies, not mutable simulator views.
    return {"tick": int(sim.tick[0]), "learner_side": side, "players": players}  # Local: events after this transition are appended by the caller.
