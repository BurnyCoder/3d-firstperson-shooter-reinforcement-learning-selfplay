"""Local: duel outcomes and registered rewards; global: shape learning without changing victory.

Engineering coefficients: docs/duel-protocol.md. Potential difference follows Ng et al.:
https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf . The identity applies
to discounted trajectories; this is not a convergence guarantee for partially observed self-play.
"""

import numpy as np  # Local: exact batched accounting independent of the learner library.

from .constants import DT, EYE_HEIGHT, PLAYER_HEIGHT  # Local: body-centre aim uses physical hitbox geometry.
from .geometry import aim_directions  # Local: use canonical pitch/yaw interpretation.
from .observation import visible_players  # Local: hidden targets cannot supply shaping information.


def aim_potential(sim):
    """Local: bounded visible aim potential; global: terminal/hidden potentials must be zero."""
    visible, relative = visible_players(sim)  # Local: FOV and wall obstruction agree with observations.
    target = relative[:, np.arange(2), [1, 0]].copy()  # Local: Stage A has exactly one enemy per observer.
    target[..., 1] += PLAYER_HEIGHT / 2 - EYE_HEIGHT  # Local: offset eye-to-eye vectors toward body centre.
    cosine = (target * aim_directions(sim.yaw, sim.pitch)).sum(-1) / np.maximum(np.linalg.norm(target, axis=-1), 1e-8)
    return .1 * np.maximum(cosine, 0) * visible[:, np.arange(2), [1, 0]] * ~sim.done[:, None]


def resolve_duel(sim, active, rewards, events, damage_before, potential_before):
    """Local: score elimination then timeout; global: simultaneous deaths draw even on the last tick."""
    living = sim.alive  # Local: combat has already resolved simultaneous health changes.
    ended = active & ((~living).any(-1) | (sim.tick * DT >= sim.config.round_seconds))
    sim.done[ended] = True  # Local: terminal state remains frozen until selective reset.
    for arena in np.flatnonzero(ended):
        survivors = np.flatnonzero(living[arena])  # Local: one survivor wins; zero/two draw.
        winner = int(survivors[0]) if len(survivors) == 1 else -1
        sim.winner[arena] = winner  # Local: elimination always precedes timeout classification.
        if winner >= 0:
            rewards[arena, winner], rewards[arena, 1 - winner] = 1, -1  # Local: terminal reward is zero-sum.
        events.append({"arena": int(arena), "type": "round_end", "winner": winner,
                       "reason": "elimination" if len(survivors) < 2 else "timeout"})
    outcome = rewards.copy()  # Local: retain unshaped terminal outcomes for diagnostic logs.
    damage = sim.damage_dealt - damage_before  # Local: combat records effective damage, capped at remaining HP.
    alpha = .25 if sim.reward_id == "R0" else .5  # Local: registered engineering choices, not fitted online.
    damage_reward = alpha * (damage - damage[:, ::-1]) / 100  # Local: exact symmetric damage difference.
    potential = (.999 * aim_potential(sim) - potential_before) * active[:, None] if sim.reward_id == "R2" else np.zeros_like(rewards)
    rewards += damage_reward + potential  # Local: R2 intentionally has separate per-player potentials.
    sim.reward_components = {"outcome": outcome, "damage": damage_reward, "potential": potential}
