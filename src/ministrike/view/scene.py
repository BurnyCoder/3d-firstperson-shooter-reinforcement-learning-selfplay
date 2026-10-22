"""Local: draw original industrial geometry; global: display the canonical simulation without physics.

Entity/camera patterns: https://github.com/pokepetter/ursina/blob/master/samples/fps.py
Eight-bit colors: https://github.com/pokepetter/ursina/blob/master/ursina/color.py
Only the shared map defines solid obstacles; every extra shape here is cosmetic.
"""

import math  # Local: translate simulation radians into Ursina's degree rotations.

import numpy as np  # Local: read map extents and copy simulation coordinates without modifying them.
from ursina import AmbientLight, DirectionalLight, Entity, Text, Vec3, camera, color, scene  # Local: graphics only.
from ursina.shaders import lit_with_shadows_shader, unlit_shader  # Local: light world geometry, keep UI readable.

from ministrike.sim.constants import EYE_HEIGHT  # Local: align the first-person camera with hitscan origin.
from ministrike.sim.specs import match_spec  # Local: renderer reads the same versioned geometry as physics.

ATTACK = color.rgb32(242, 158, 79)  # Local: warm attackers; global: consistent team color throughout the viewer.
DEFEND = color.rgb32(76, 202, 190)  # Local: cool defenders; global: distinguish teams without copied game assets.
INK = color.rgb32(17, 26, 35)  # Local: dark navy outlines; global: provide readable industrial contrast.


def block(position, scale, tint, parent=scene, **kwargs):
    """Local: create a cosmetic box; global: all collision and shooting remain in Simulation."""
    return Entity(  # Local: use the built-in cube so no external asset download is necessary.
        parent=parent, model="cube", position=position, scale=scale, color=tint,
        shader=lit_with_shadows_shader, **kwargs,
    )  # Local: intentionally omit a collider, preventing a second physics authority.


class ArenaScene:
    """Local: construct and synchronize visible entities; global: presentation never mutates match state."""

    def __init__(self, spec=None):
        """Local: build floor, cover, sites, players and rifle; global: reuse one compact original map."""
        self.spec = spec or match_spec()  # Local: legacy caller defaults remain available.
        self.walls, self.sites, _ = self.spec.geometry()  # Local: no independently drawn gameplay geometry.
        self.low = np.min(self.walls[:, :3], axis=0)  # Local: derive the envelope from canonical map bounds.
        self.high = np.max(self.walls[:, 3:], axis=0)  # Local: no hard-coded physics/render offsets.
        self.center = (self.low + self.high) / 2  # Local: place the overview camera relative to this map.
        span = max(self.high[0] - self.low[0], self.high[2] - self.low[2])  # Local: fit the initial free camera to this map.
        self.free_position = np.array([self.center[0], span * 0.94, self.center[2] - span * 0.72], dtype=np.float32)
        self.free_rotation = np.array([math.degrees(math.atan2(0.94, 0.72)), 0, 0], dtype=np.float32)  # Local: downward view.
        self.players = []  # Local: retain one visible entity per permanent player slot.
        self.flashes = []  # Local: muzzle effects depend only on shots reported by the simulation.
        self.flash_until = np.zeros(self.spec.players)  # Local: cosmetic lifetimes use no simulation randomness.
        self._build_map()  # Local: construct static geometry before attaching dynamic actors.
        self._build_players()  # Local: create original block-figure operators for each team.
        self.bomb = block((0, 0.16, 0), (0.42, 0.28, 0.3), ATTACK)  # Local: render a small objective case.
        block((0, 0.55, 0), (0.75, 0.12, 0.55), INK, parent=self.bomb)  # Local: add the case's top strap.
        self.rifle = Entity(parent=camera, position=(0.30, -0.23, 0.58))  # Local: camera-relative FPS viewmodel.
        block((0, 0, 0), (0.11, 0.14, 0.55), INK, parent=self.rifle)  # Local: receiver geometry.
        block((0, 0.01, 0.43), (0.04, 0.05, 0.4), color.rgb32(78, 91, 100), parent=self.rifle)  # Local: barrel.
        block((0, -0.13, 0.05), (0.08, 0.2, 0.14), color.rgb32(50, 67, 76), parent=self.rifle)  # Local: magazine.
        block((0, 0.095, 0.02), (0.045, 0.04, 0.09), ATTACK, parent=self.rifle)  # Local: readable top sight.
        self.first_flash = Entity(  # Local: display the human's confirmed shot without calculating a hit.
            parent=self.rifle, model="sphere", position=(0, 0, 0.65), scale=(0.16, 0.16, 0.28),
            color=color.rgb32(255, 222, 129), shader=unlit_shader, enabled=False,
        )  # Local: the short lifetime is managed by the render clock below.
        AmbientLight(color=color.rgba32(170, 184, 196, 255))  # Local: illuminate cover faces in the shadowed lanes.
        sun = DirectionalLight(shadows=True)  # Local: shadows communicate obstacle shapes and player locations.
        sun.look_at(Vec3(0.7, -1.4, 0.5))  # Local: fixed direction keeps rendering deterministic.

    def _build_map(self):
        """Local: render map boxes and floor markings; global: preserve exact visual/physical wall parity."""
        width, depth = self.high[0] - self.low[0], self.high[2] - self.low[2]  # Local: canonical floor extent.
        block((self.center[0], -0.18, self.center[2]), (width, 0.3, depth), color.rgb32(60, 76, 86))  # Local: floor.
        for x in np.arange(self.low[0], self.high[0], 4):  # Local: repeated floor seams reveal motion and scale.
            block((x, -0.021, self.center[2]), (0.025, 0.01, depth), color.rgb32(83, 98, 105))  # Local: thin seam.
        for z in np.arange(self.low[2], self.high[2], 4):  # Local: complete the regular floor tile grid.
            block((self.center[0], -0.021, z), (width, 0.01, 0.025), color.rgb32(83, 98, 105))  # Local: thin seam.
        for wall in self.walls:  # Local: draw exactly one primary block per collision AABB.
            midpoint, size = (wall[:3] + wall[3:]) / 2, wall[3:] - wall[:3]  # Local: convert bounds to cubes.
            tint = color.rgb32(94, 115, 127) if size[1] > 2 else color.rgb32(109, 126, 125)  # Local: cover contrast.
            block(tuple(midpoint), tuple(size), tint)  # Local: display the complete physical obstacle.
            cap = midpoint.copy()  # Local: copy prevents render decorations from altering canonical map data.
            cap[1] = wall[4] + 0.035  # Local: position a small industrial cap just above the obstacle.
            block(tuple(cap), (size[0], 0.07, size[2]), color.rgb32(38, 54, 66))  # Local: highlight top edges.
            if size[1] < 2:  # Local: low cover gets a visually distinct safety stripe.
                stripe = midpoint.copy()  # Local: maintain immutable obstacle data while positioning the stripe.
                stripe[1] = wall[4] - 0.12  # Local: keep the decorative stripe on the upper cover face.
                block(tuple(stripe), (size[0] + 0.012, 0.10, size[2] + 0.012), ATTACK)  # Local: amber band.
        for label, site in zip(("A", "B")[:len(self.sites)], self.sites, strict=True):  # Local: duel has no sites.
            block((site[0], 0.003, site[2]), (3.8, 0.014, 3.8), color.rgb32(109, 83, 52))  # Local: painted zone.
            for offset in (-1.9, 1.9):  # Local: outline the planted-objective area with clear amber corners.
                block((site[0] + offset, 0.015, site[2]), (0.08, 0.025, 3.8), ATTACK)  # Local: vertical edges.
                block((site[0], 0.015, site[2] + offset), (3.8, 0.025, 0.08), ATTACK)  # Local: horizontal edges.
            Text(  # Local: float site labels above lanes; global: help players orient on the original map.
                text=label, parent=scene, position=(site[0], 3.2, site[2]), scale=3,
                origin=(0, 0), billboard=True, color=ATTACK,
            )  # Local: labels contain no hidden dynamic game information.

    def _build_players(self):
        """Local: create compact operator silhouettes; global: make team, heading, and death readable."""
        for index in range(self.spec.players):  # Local: retain stable slots even after an operator dies.
            tint = ATTACK if index < self.spec.team_size else DEFEND  # Local: canonical team allocation.
            root = Entity()  # Local: moving this parent moves cosmetic anatomy together.
            block((0, 0.94, 0), (0.56, 0.72, 0.35), tint, parent=root)  # Local: colored torso.
            block((0, 0.94, 0.19), (0.36, 0.42, 0.08), INK, parent=root)  # Local: front armor indicates heading.
            Entity(parent=root, model="sphere", position=(0, 1.54, 0), scale=0.43, color=tint,
                   shader=lit_with_shadows_shader)  # Local: helmet follows the simulation's 1.8 m silhouette.
            block((0, 1.55, 0.195), (0.30, 0.09, 0.07), INK, parent=root)  # Local: visor faces the aiming direction.
            for side in (-1, 1):  # Local: build both legs with the same compact geometry.
                block((side * 0.16, 0.34, 0), (0.22, 0.64, 0.25), INK, parent=root)  # Local: boots/trousers.
                block((side * 0.34, 1.0, 0.05), (0.16, 0.46, 0.19), tint, parent=root)  # Local: arms.
            block((0.26, 1.16, 0.42), (0.13, 0.13, 0.75), INK, parent=root)  # Local: forward-pointing rifle.
            flash = Entity(parent=root, model="sphere", position=(0.26, 1.16, 0.87), scale=0.2,
                           color=color.rgb32(255, 220, 128), shader=unlit_shader, enabled=False)  # Local: flash.
            self.players.append(root)  # Local: maintain direct player-slot indexing for synchronization.
            self.flashes.append(flash)  # Local: avoid searching scene objects every frame.

    def register_shots(self, previous_ammo, current_ammo, now):
        """Local: schedule muzzle effects from ammunition loss; global: graphics cannot invent a valid shot."""
        fired = current_ammo < previous_ammo  # Local: only confirmed cartridge consumption creates a flash.
        self.flash_until[fired] = now + 0.075  # Local: cosmetic flash duration never feeds back into the sim.

    def move_free_camera(self, elapsed, keys, mouse_delta):
        """Local: fly a detached spectator camera; global: its position never enters observations or simulation physics."""
        self.free_rotation[0] = np.clip(self.free_rotation[0] - mouse_delta[1] * 90, -85, 85)  # Local: RMB vertical look.
        self.free_rotation[1] += mouse_delta[0] * 90  # Local: RMB horizontal look rotates the cosmetic camera only.
        pitch, yaw = np.deg2rad(self.free_rotation[:2])  # Local: convert camera degree rotations into direction vectors.
        forward = np.array([math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        right = np.array([math.cos(yaw), 0, -math.sin(yaw)])  # Local: strafe horizontally relative to the camera heading.
        movement = forward * (keys.get("w", 0) - keys.get("s", 0)) + right * (keys.get("d", 0) - keys.get("a", 0))
        movement[1] += keys.get("e", 0) - keys.get("q", 0)  # Local: Q/E descend/ascend without binding takeover's F key.
        movement /= max(1.0, float(np.linalg.norm(movement)))  # Local: diagonal movement does not speed up the camera.
        self.free_position += movement * elapsed * (20 if keys.get("shift", 0) else 10)  # Local: optional faster travel.

    def sync(self, simulation, followed: int | None, now: float):
        """Local: copy match state to graphics; global: read-only presentation supports headless parity."""
        for index, entity in enumerate(self.players):  # Local: synchronize all permanent player slots.
            entity.position = tuple(simulation.positions[0, index])  # Local: assign a tuple, never a mutable view.
            entity.rotation_y = math.degrees(float(simulation.yaw[0, index]))  # Local: simulation yaw maps to +Y.
            entity.rotation_z = 0 if simulation.alive[0, index] else 90  # Local: dead operators visibly fall over.
            entity.enabled = index != followed  # Local: hide the body that would obstruct the first-person camera.
            self.flashes[index].enabled = now < self.flash_until[index] and simulation.alive[0, index]  # Local: flash.
        self.bomb.position = tuple(simulation.bomb_position[0] + np.asarray([0, 0.15, 0]))  # Local: objective case.
        self.bomb.enabled = self.spec.bomb and (int(simulation.bomb_carrier[0]) < 0 or bool(simulation.bomb_planted[0]))
        self.bomb.color = color.rgb32(255, 91, 65) if simulation.bomb_planted[0] else ATTACK  # Local: armed warning.
        self.rifle.enabled = followed is not None and bool(simulation.alive[0, followed])  # Local: FPS-only viewmodel.
        self.first_flash.enabled = followed is not None and now < self.flash_until[followed]  # Local: confirmed flash.
        if followed is not None:  # Local: first-person and follow modes use exactly the actor's aiming direction.
            point = simulation.positions[0, followed].copy()  # Local: protect authoritative position arrays.
            point[1] += EYE_HEIGHT  # Local: align camera height with the simulation's hitscan eye origin.
            camera.position = tuple(point)  # Local: do not use a FirstPersonController or a graphics collider.
            camera.rotation = (-math.degrees(float(simulation.pitch[0, followed])),
                               math.degrees(float(simulation.yaw[0, followed])), 0)  # Local: positive pitch looks up.
        else:  # Local: the free spectator camera begins in overview but can fly anywhere without affecting the arena.
            camera.position = tuple(self.free_position)  # Local: camera travel never writes simulation positions.
            camera.rotation = tuple(self.free_rotation)  # Local: cosmetic yaw/pitch stay separate from every bot's aim.
