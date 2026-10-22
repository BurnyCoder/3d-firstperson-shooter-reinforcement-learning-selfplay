"""Local: visibility-filtered local observations. Global: actor inputs never expose hidden enemy dynamics.

Static map probes and known bomb sites are public. Enemy health is never exposed, even when visible.
Only the observing team's own carrier communication can reveal an unplanted hidden carrier location.
"""

import numpy as np  # Local: batch observers, targets, geometry, and independent arenas.

from .constants import DT, EYE_HEIGHT, RAY_ANGLES  # Local: shared feature layout and ray origin.
from .geometry import aim_directions, ray_boxes  # Local: visibility and weapons agree on solid geometry.


def visible_players(sim):
    """Local: calculate a 3D FOV/LOS mask. Global: policies see only currently observable opponents."""
    eyes = sim.positions + np.array([0, EYE_HEIGHT, 0], dtype=np.float32)  # Local: all sight rays begin at eye height.
    relative = eyes[:, None, :, :] - eyes[:, :, None, :]  # Local: observer-to-target vectors for all fixed slots.
    distance = np.linalg.norm(relative, axis=-1)  # Local: range is needed to stop LOS rays at each target.
    direction = relative / np.maximum(distance[..., None], 1e-8)  # Local: avoid undefined direction for own slot.
    facing = aim_directions(sim.yaw, sim.pitch)  # Local: the visible cone follows actual aim orientation.
    in_fov = (direction * facing[:, :, None]).sum(-1) >= np.cos(np.deg2rad(sim.config.fov_degrees / 2))
    wall_distance = ray_boxes(eyes[:, :, None, None], direction[:, :, :, None], sim.walls[:, :3], sim.walls[:, 3:]).min(-1)
    visible = in_fov & (wall_distance >= distance - 1e-5) & sim.alive[:, None, :] & sim.alive[:, :, None]
    visible |= np.eye(sim.players, dtype=bool)[None] & sim.alive[:, :, None]  # Local: a living observer always knows itself.
    return visible, relative  # Local: return geometry once for reuse in observation construction.


def make_frame(sim):
    """Local: construct normalized float32 features. Global: callable without RNG or state mutation."""
    frame = np.zeros((sim.num_envs, sim.players, sim.spec.frame_dim), dtype=np.float32)  # Local: hidden defaults zero.
    visible, relative = visible_players(sim)  # Local: one FOV/LOS calculation serves all player features.
    teams = np.arange(sim.players) // sim.team_size  # Local: identities remain public and stable after death.
    frame[..., :3] = sim.positions / 30  # Local: known own world position supports static-map navigation.
    frame[..., 3:7] = np.stack((np.sin(sim.yaw), np.cos(sim.yaw), np.sin(sim.pitch), np.cos(sim.pitch)), -1)
    frame[..., 7] = sim.health / sim.config.health  # Local: own health is observable.
    frame[..., 8] = sim.ammo / sim.config.magazine  # Local: own rifle ammunition is observable.
    frame[..., 9] = sim.reserve / max(sim.config.reserve_ammo, 1)  # Local: normalize finite spare ammunition.
    frame[..., 10] = sim.reload_timer / sim.config.reload_seconds  # Local: expose own reload completion fraction.
    frame[..., 11] = sim.alive  # Local: stable dead slots communicate inability to act.
    frame[..., 12] = teams  # Local: one shared actor can condition on attack versus defense.
    frame[..., 13] = np.arange(sim.players)[None] == sim.bomb_carrier[:, None]  # Local: own objective possession.
    frame[..., 14] = np.maximum(1 - sim.tick[:, None] * DT / sim.config.round_seconds, 0)  # Local: public round clock.
    frame[..., 15] = sim.interaction / max(sim.config.defuse_seconds, sim.config.plant_seconds)  # Local: own progress.
    carrier = np.maximum(sim.bomb_carrier, 0)  # Local: safe lookup before checking valid carrier sentinel.
    seen_carrier = visible[np.arange(sim.num_envs)[:, None], np.arange(sim.players)[None], carrier[:, None]]
    known = sim.bomb_planted[:, None] | (teams[None] == 0) | (seen_carrier & (sim.bomb_carrier[:, None] >= 0))
    frame[..., 16:19] = (sim.bomb_position[:, None] - sim.positions) / 30 * known[..., None]
    frame[..., 19] = sim.bomb_planted[:, None]  # Local: plant announcements make the planted state public.
    frame[..., 20] = sim.bomb_timer[:, None] / sim.config.bomb_seconds * sim.bomb_planted[:, None]
    frame[..., 21] = known  # Local: distinguish hidden bomb zeros from an actual colocated bomb.
    if sim.spec.bomb:
        frame[..., 22:28] = ((sim.sites[None, None] - sim.positions[:, :, None]) / 30).reshape(sim.num_envs, sim.players, 6)
    else:
        frame[..., 13] = 0  # Local: disabled objective fields cannot create fictitious bomb information.
        frame[..., 15:28] = 0  # Local: keep the shared field offsets while explicitly masking inapplicable features.
    probes = sim.positions + np.array([0, .5, 0], dtype=np.float32)  # Local: waist-level probes detect impassable low cover.
    ray_yaw = sim.yaw[..., None] + RAY_ANGLES  # Local: probes rotate with the observer's orientation.
    rays = aim_directions(ray_yaw, np.zeros_like(ray_yaw))  # Local: horizontal rays provide nearby map shape.
    ranges = ray_boxes(probes[:, :, None, None], rays[:, :, :, None], sim.walls[:, :3], sim.walls[:, 3:]).min(-1)
    frame[..., 28:36] = np.minimum(ranges, 30) / 30  # Local: bounded static geometry distances.
    players = frame[..., 36:].reshape(sim.num_envs, sim.players, sim.players, 6)  # Local: stable target slots.
    same_team = teams[:, None] == teams[None, :]  # Local: team identities do not reveal enemy dynamic state.
    players[..., :3] = relative / 30 * visible[..., None]  # Local: hidden targets have zero relative position.
    players[..., 3] = visible  # Local: indicate which target rows currently contain measurements.
    players[..., 4] = same_team  # Local: team metadata remains public independent of visibility.
    players[..., 5] = sim.health[:, None] / sim.config.health * same_team * visible  # Local: enemy HP always zero.
    return frame  # Local: callers own the new array and cannot mutate simulator state through it.
