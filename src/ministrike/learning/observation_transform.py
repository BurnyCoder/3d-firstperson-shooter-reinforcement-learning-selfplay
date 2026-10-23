"""Local: versioned actor-only coordinate transform; global: preserve the public observation and action contracts.

Relative features are motivated by Unity ML-Agents' observation guidance:
https://unity-technologies.github.io/ml-agents/Learning-Environment-Design-Agents/#vector-observation-summary-best-practices
The particular rotation is an engineering intervention, not a claim of proven superiority.
It applies an orthonormal basis from the game's sim.geometry.aim_directions convention;
https://docs.pytorch.org/docs/2.14/generated/torch.stack.html supplies tensor assembly.
No hidden state, target selection, aim correction, RNG, or trainable parameter is introduced.
"""

import torch  # Local: use the actor's existing device/dtype; no NumPy round trip during inference.

from ministrike.sim.constants import HISTORY_LENGTH  # Local: transform each historical frame using its own pose.

TRANSFORMS = ("identity", "egocentric-v1")  # Local: unknown checkpoint representations fail closed.


def transform_observation(observation, identity):
    """Local: rotate already-masked relative XYZ into right/up/forward; global: actions are still entirely learned."""
    if identity not in TRANSFORMS:
        raise ValueError(f"Unknown observation transform: {identity}")
    if identity == "identity":
        return observation  # Local: preserve old behavior and all bits without allocating a second tensor.
    frames = observation.reshape(*observation.shape[:-1], HISTORY_LENGTH, -1)
    output = frames.clone()  # Local: the source belongs to the collector and must never be mutated.
    players = frames[..., 36:].reshape(*frames.shape[:-1], -1, 6)
    target = output[..., 36:].reshape_as(players)
    x, y, z = players[..., 0], players[..., 1], players[..., 2]
    sy, cy, sp, cp = [frames[..., index, None] for index in (3, 4, 5, 6)]  # Local: supplied sine/cosine values, not hidden world pose.
    horizontal = sy * x + cy * z  # Local: project the world displacement onto horizontal facing.
    target[..., :3] = torch.stack((cy * x - sy * z, cp * y - sp * horizontal, sp * y + cp * horizontal), -1)
    return output.flatten(-2)  # Local: only coordinate values change; masks, dimensions, history and scales stay fixed.
