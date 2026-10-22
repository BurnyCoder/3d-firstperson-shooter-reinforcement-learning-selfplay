"""Local: original compact map geometry. Global: one source supplies collisions, visibility and rendering.

Coordinates are metres, Y points up, and Z points toward the defending spawn.
No player-player collision is modeled; walls are solid axis-aligned boxes.
"""

import numpy as np  # Local: immutable geometry arrays support vectorized intersection tests.

WALLS = np.array([  # Local: each row is min XYZ then max XYZ, following PBRT Bounds3 conventions.
    [-15.5, 0, -12.5, -15, 3.2, 12.5],  # Local: west perimeter.
    [15, 0, -12.5, 15.5, 3.2, 12.5],  # Local: east perimeter.
    [-15.5, 0, -12.5, 15.5, 3.2, -12],  # Local: attack-side perimeter.
    [-15.5, 0, 12, 15.5, 3.2, 12.5],  # Local: defense-side perimeter.
    [-11, 0, -7.5, 11, 3.2, -7],  # Local: attack spawn baffle prevents cross-map kills before players reach either outer route.
    [-11, 0, 7, 11, 3.2, 7.5],  # Local: mirrored defense baffle protects starts while keeping both bomb sites reachable behind it.
    [-2, 0, -4, 2, 3.2, 4],  # Local: central building separates the two outer attack routes.
    [-11, 0, -2, -7, 1.0, -1],  # Local: low western cover blocks bodies while allowing eye-height sightlines.
    [7, 0, -2, 11, 1.0, -1],  # Local: mirrored eastern low cover.
    [-11, 0, 4, -9, 2.4, 5.0],  # Local: western objective approach has a tall sight blocker.
    [9, 0, 4, 11, 2.4, 5.0],  # Local: eastern objective approach has mirrored tall cover.
], dtype=np.float32)  # Local: contiguous float32 data matches batched simulation storage.
SITES = np.array([[-8, 0, 8], [8, 0, 8]], dtype=np.float32)  # Local: two legal bomb sites.
SPAWNS = np.array([[-6, 0, -10], [-3, 0, -10], [0, 0, -10], [3, 0, -10], [6, 0, -10],
                   [-6, 0, 10], [-3, 0, 10], [0, 0, 10], [3, 0, 10], [6, 0, 10]], dtype=np.float32)
WALLS.flags.writeable = False  # Local: renderers cannot mutate authoritative geometry.
SITES.flags.writeable = False  # Local: objective locations are shared constants.
SPAWNS.flags.writeable = False  # Local: round starts cannot corrupt the template.
