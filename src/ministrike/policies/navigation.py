"""Local: shortest paths through the public box map. Global: scripted benchmark navigation only.

Sources: https://docs.python.org/3/library/collections.html#collections.deque
and https://docs.python.org/3/library/functools.html#functools.lru_cache
No trained policy calls this module; it is an explicit hand-written control condition.
"""

from collections import deque  # Breadth-first search exactly solves unit-cost grid distances.
from functools import lru_cache  # Reuse distance fields for static objective cells.

import numpy as np  # Vectorize per-player lookup after computing the small map field.

from ministrike.sim.map import WALLS  # Derive navigation obstacles from authoritative geometry.

OFFSETS = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)  # Cardinal edges avoid corner cutting.


@lru_cache(maxsize=1)
def walkable() -> np.ndarray:
    """Local: rasterize inflated walls. Global: scripted players respect their physical radius."""
    x, z = np.meshgrid(np.arange(30) - 14.5, np.arange(24) - 11.5, indexing="ij")  # One-metre cell centers.
    blocked = np.zeros((30, 24), dtype=bool)  # Begin with open floor cells.
    for wall in WALLS:  # Map size is small; this loop executes once per process.
        blocked |= (x >= wall[0] - 0.4) & (x <= wall[3] + 0.4) & (z >= wall[2] - 0.4) & (z <= wall[5] + 0.4)  # Inflate by body radius.
    return ~blocked  # All routing decisions use only publicly known geometry.


@lru_cache(maxsize=128)
def distances(goal_x: int, goal_z: int) -> np.ndarray:
    """Local: cache a BFS distance field. Global: no enemy positions enter navigation."""
    valid = walkable()  # Share the same raster across every goal.
    goal = np.array([goal_x, goal_z])  # Clamp or repair a target that falls inside geometry.
    cells = np.argwhere(valid)  # Candidate nearest reachable floor cells.
    goal = cells[np.argmin(np.sum((cells - goal) ** 2, axis=1))]  # Snap bomb/site position to free floor.
    field = np.full(valid.shape, np.inf, dtype=np.float32)  # Infinity records cells disconnected from the target.
    field[tuple(goal)] = 0  # BFS starts at the goal and expands outwards.
    pending = deque([tuple(goal)])  # FIFO order visits minimum edge counts first.
    while pending:  # Each free cell enters the queue at most once.
        current = pending.popleft()  # Pop the next nearest discovered cell.
        for offset in OFFSETS[1:]:  # Four-connected movement cannot cross blocked corners.
            neighbor = np.asarray(current) + offset  # Compute an adjacent grid coordinate.
            if np.any(neighbor < 0) or neighbor[0] >= 30 or neighbor[1] >= 24:  # Reject coordinates outside the map.
                continue  # No array lookup occurs for an invalid cell.
            key = tuple(neighbor)  # NumPy uses coordinate tuples for scalar indexing.
            if valid[key] and np.isinf(field[key]):  # Enqueue only first-time reachable visits.
                field[key] = field[current] + 1  # Unit edge costs make BFS exact on this grid.
                pending.append(key)  # Later iterations expand this frontier.
    return field  # Cached fields are treated as immutable by callers.


def next_waypoint(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Local: choose the downhill neighboring cell. Global: route scripted baselines around cover."""
    position_cells = np.clip(np.floor(positions[:, [0, 2]] + [15, 12]).astype(int), [0, 0], [29, 23])  # Map positions to indices.
    goal_cells = np.clip(np.floor(targets[:, [0, 2]] + [15, 12]).astype(int), [0, 0], [29, 23])  # Quantize goals for cache reuse.
    result = targets.copy()  # Preserve exact target positions once a player reaches its cell.
    for goal in np.unique(goal_cells, axis=0):  # Usually only two site fields are needed for a whole batch.
        selected = np.all(goal_cells == goal, axis=1)  # Gather players sharing this navigation target.
        current = position_cells[selected]  # Keep the player batch axis intact.
        candidates = np.clip(current[:, None] + OFFSETS, [0, 0], [29, 23])  # Safe neighboring lookup coordinates.
        field = distances(int(goal[0]), int(goal[1]))  # Reuse the immutable BFS result.
        best = candidates[np.arange(len(current)), np.argmin(field[candidates[..., 0], candidates[..., 1]], axis=1)]  # Follow decreasing distance.
        advancing = np.any(current != goal, axis=1)  # Final approach uses the actual objective rather than a cell center.
        indices = np.flatnonzero(selected)[advancing]  # Scatter only players still navigating.
        result[indices, 0] = best[advancing, 0] - 14.5  # Convert back to world X.
        result[indices, 2] = best[advancing, 1] - 11.5  # Convert back to world Z.
    return result  # No controller state or random generator is modified.
