"""Local: vectorized geometric primitives. Global: movement, aiming and vision agree on obstruction.

Slab intersection follows https://www.pbr-book.org/4ed/Geometry_and_Transformations/Bounding_Boxes.
Parallel rays are handled explicitly to avoid divide-by-zero and NaN boundary cases.
"""

import numpy as np  # Local: broadcasting batches independent ray/box pairs.


def ray_boxes(origins, directions, lower, upper):
    """Local: nearest nonnegative ray entry distance per box. Global: shared hitscan and LOS primitive."""
    parallel = np.abs(directions) < 1e-8  # Local: a parallel ray cannot enter through that axis's planes.
    inverse = np.divide(1.0, directions, out=np.zeros_like(directions), where=~parallel)  # Local: safe reciprocals.
    first, second = (lower - origins) * inverse, (upper - origins) * inverse  # Local: two plane crossings per axis.
    near = np.where(parallel, -np.inf, np.minimum(first, second))  # Local: parallel axes do not bound entry time.
    far = np.where(parallel, np.inf, np.maximum(first, second))  # Local: parallel axes do not bound exit time.
    outside = np.any(parallel & ((origins < lower) | (origins > upper)), axis=-1)  # Local: parallel outside misses.
    entry = np.maximum(np.max(near, axis=-1), 0)  # Local: rays starting inside a box hit at the origin.
    exit_time = np.min(far, axis=-1)  # Local: earliest exit bounds the common slab interval.
    return np.where((exit_time >= entry) & ~outside, entry, np.inf)  # Local: infinity expresses a miss.


def aim_directions(yaw, pitch):
    """Local: angular state becomes unit vectors. Global: the renderer and rifle use the same camera convention."""
    return np.stack((np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)), axis=-1)
