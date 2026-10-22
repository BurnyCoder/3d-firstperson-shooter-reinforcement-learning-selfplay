"""Local: movement and weapon clocks. Global: all controllers obey identical fixed-tick mechanics."""

import numpy as np  # Local: vectorize all players and independent arenas.

from .constants import DT, PLAYER_RADIUS  # Local: common timing and collision dimensions.


def advance_movement(sim, controls, active):
    """Local: rotate and slide living players against boxes. Global: no render-dependent physics."""
    movable = sim.alive & active[:, None]  # Local: dead and terminal slots remain fixed.
    walls = sim.walls  # Local: the renderer and physics read the same selected map geometry.
    sim.yaw[:] = np.where(movable, (sim.yaw + controls[..., 2] + np.pi) % (2 * np.pi) - np.pi, sim.yaw)
    sim.pitch[:] = np.where(movable, np.clip(sim.pitch + controls[..., 3], -1.35, 1.35), sim.pitch)
    local = np.clip(controls[..., :2], -1, 1)  # Local: clamp controller movement to supported speeds.
    local /= np.maximum(np.linalg.norm(local, axis=-1, keepdims=True), 1)  # Local: diagonal motion stays normalized.
    forward, strafe = local[..., 0], local[..., 1]  # Local: unpack view-relative movement axes.
    dx = (forward * np.sin(sim.yaw) + strafe * np.cos(sim.yaw)) * sim.config.movement_speed * DT
    dz = (forward * np.cos(sim.yaw) - strafe * np.sin(sim.yaw)) * sim.config.movement_speed * DT
    for axis, displacement in ((0, dx), (2, dz)):  # Local: separate axes permit sliding along walls.
        candidate = sim.positions.copy()  # Local: tentative positions do not modify blocked actors.
        candidate[..., axis] += displacement * movable  # Local: keep terminal/dead actors stationary.
        x, z = candidate[..., 0, None], candidate[..., 2, None]  # Local: add wall broadcasting dimension.
        blocked = np.any((x > walls[:, 0] - PLAYER_RADIUS) & (x < walls[:, 3] + PLAYER_RADIUS)
                         & (z > walls[:, 2] - PLAYER_RADIUS) & (z < walls[:, 5] + PLAYER_RADIUS), axis=-1)
        sim.positions[..., axis] = np.where(blocked, sim.positions[..., axis], candidate[..., axis])
    running_cooldown = np.where(sim.cooldown > 0, sim.cooldown - DT, 0)  # Local: preserve one tick's overshoot for exact average cadence.
    sim.cooldown[:] = np.where(active[:, None], running_cooldown, sim.cooldown)  # Local: terminal clocks stay frozen.
    reloading = sim.reload_timer > 0  # Local: preserve previous state to identify timer completion.
    sim.reload_timer[:] = np.where(active[:, None], np.maximum(sim.reload_timer - DT, 0), sim.reload_timer)
    completed = reloading & (sim.reload_timer < 1e-6) & movable  # Local: dead players cannot reload.
    transfer = np.minimum(sim.config.magazine - sim.ammo, sim.reserve) * completed  # Local: ammo conservation.
    sim.ammo += transfer  # Local: refill only completed magazines.
    sim.reserve -= transfer  # Local: spare ammunition is finite.
    start_reload = (controls[..., 5] > .5) & movable & (sim.reload_timer <= 0) & (sim.reserve > 0)
    start_reload &= sim.ammo < sim.config.magazine  # Local: full magazines cannot restart a reload.
    sim.reload_timer[start_reload] = sim.config.reload_seconds  # Local: held input starts once while timer runs.
