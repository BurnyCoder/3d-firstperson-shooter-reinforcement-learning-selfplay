"""Local: simultaneous, nearest-hit rifle resolution. Global: training and play have identical shooting rules."""

import numpy as np  # Local: batch rays and damage accumulation across independent arenas.

from .constants import EYE_HEIGHT, PLAYER_HEIGHT, PLAYER_RADIUS  # Local: consistent human/model hit geometry.
from .geometry import aim_directions, ray_boxes  # Local: shared slab intersection from cited geometry module.
from .map import WALLS  # Local: the visible wall geometry also blocks bullets.


def resolve_combat(sim, controls, active, rewards, events):
    """Local: collect all shots before applying damage. Global: no advantage from player iteration order."""
    firing = (controls[..., 4] > .5) & sim.alive & active[:, None]  # Local: only live active slots can act.
    firing &= (sim.ammo > 0) & (sim.cooldown < 1e-6) & (sim.reload_timer < 1e-6)  # Local: weapon constraints.
    if not firing.any():  # Local: avoid expensive ray work on idle ticks.
        return  # Local: authoritative state remains unchanged by nonexistent shots.
    origins = sim.positions + np.array([0, EYE_HEIGHT, 0], dtype=np.float32)  # Local: fire from eye height.
    directions = aim_directions(sim.yaw, sim.pitch)  # Local: physics uses exactly the controller's orientation.
    wall_distance = ray_boxes(origins[:, :, None], directions[:, :, None], WALLS[:, :3], WALLS[:, 3:]).min(-1)
    lower = sim.positions - np.array([PLAYER_RADIUS, 0, PLAYER_RADIUS], dtype=np.float32)  # Local: target feet.
    upper = sim.positions + np.array([PLAYER_RADIUS, PLAYER_HEIGHT, PLAYER_RADIUS], dtype=np.float32)
    target_distance = ray_boxes(origins[:, :, None], directions[:, :, None], lower[:, None], upper[:, None])
    target_distance = np.where(sim.alive[:, None, :] & ~np.eye(10, dtype=bool)[None], target_distance, np.inf)
    target = target_distance.argmin(-1)  # Local: teammates can physically intercept a shot.
    nearest = target_distance.min(-1)  # Local: rifle hits only the nearest occupied hitbox.
    hits = firing & (nearest < wall_distance) & np.isfinite(nearest)  # Local: walls win equal-distance ties.
    hits &= (target // 5) != (np.arange(10)[None] // 5)  # Local: friendly fire is disabled, but bodies block.
    sim.ammo -= firing  # Local: misses still consume ammunition.
    sim.cooldown[firing] += 1 / sim.config.shots_per_second  # Local: retain tick overshoot to average six shots per second.
    damage = np.zeros_like(sim.health)  # Local: aggregate every simultaneous hit before touching health.
    arenas, shooters = np.nonzero(hits)  # Local: sparse processing is cheaper for actual damage events.
    np.add.at(damage, (arenas, target[arenas, shooters]), sim.config.damage)  # Local: repeated targets accumulate.
    effective = np.minimum(damage, sim.health)  # Local: overkill does not inflate shaping or damage metrics.
    previous_alive = sim.alive.copy()  # Local: kill events compare pre- and post-damage life states.
    sim.health[:] = np.maximum(sim.health - damage, 0)  # Local: simultaneous lethal trades are preserved.
    attack_damage = effective[:, 5:].sum(-1)  # Local: damage against defenders credits attackers.
    defense_damage = effective[:, :5].sum(-1)  # Local: damage against attackers credits defenders.
    shaped = .1 * (attack_damage - defense_damage) / (5 * sim.config.health)  # Local: total per-team damage bonus≤.1.
    rewards[:, 0] += shaped  # Local: zero-sum attack-side shaping.
    rewards[:, 1] -= shaped  # Local: defense receives the exact negation.
    sim.damage_dealt[:, 0] += attack_damage  # Local: preserve raw round metrics independently of rewards.
    sim.damage_dealt[:, 1] += defense_damage  # Local: record the defending team's effective damage.
    sim.shots_fired += firing  # Local: round-level inactivity/accuracy analysis uses actual successful firing.
    for arena, shooter in zip(*np.nonzero(firing), strict=True):  # Local: matching nonzero indices identify actual shots.
        events.append({"arena": int(arena), "type": "shot", "player": int(shooter)})
    for arena, shooter in zip(arenas, shooters, strict=True):  # Local: paired hit coordinates preserve attribution.
        victim = target[arena, shooter]  # Local: each hit retains its own shooter and target identity.
        amount = float(effective[arena, victim] * sim.config.damage / damage[arena, victim])  # Local: conserve effective HP lost.
        events.append({"arena": int(arena), "type": "damage", "player": int(victim), "attacker": int(shooter),
                       "victim": int(victim), "amount": amount, "value": amount, "team": int(shooter // 5)})
    for arena, victim in zip(*np.nonzero(previous_alive & ~sim.alive), strict=True):  # Local: paired death coordinates retain slots.
        events.append({"arena": int(arena), "type": "kill", "player": int(victim), "team": int(victim // 5)})
