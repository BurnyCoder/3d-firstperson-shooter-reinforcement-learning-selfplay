"""Local: safe model/state serialization; global: reproducible public checkpoints.

Source: https://docs.pytorch.org/docs/stable/notes/serialization.html
"""

from pathlib import Path  # Local: atomic local checkpoint paths; global: recoverability.

import numpy as np  # Local: serialize simulation arrays; global: NumPy state compatibility.
import torch  # Local: tensor-only loading; global: avoid arbitrary pickle execution.

from ministrike.learning.models import ActorNetwork, FactorizedCategorical
from ministrike.sim.constants import ACTION_NVECS, OBS_DIM, decode_actions


def encode_state(value):
    """Local: replace NumPy objects with safe tensors; global: weights_only checkpoint loading."""
    if isinstance(value, np.ndarray):  # NumPy reconstructors are intentionally not allowlisted.
        return {"__ndarray__": torch.from_numpy(value.copy()), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):  # NumPy scalar objects also need primitive equivalents.
        return value.item()
    if isinstance(value, dict):  # Preserve optimizer/simulator nested structures.
        return {key: encode_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):  # Preserve Python RNG tuples and heterogeneous records.
        return type(value)(encode_state(item) for item in value)
    return value  # Tensors and Python primitives are already supported safely.


def decode_state(value):
    """Local: reconstruct safe arrays; global: restore the simulation's native representation."""
    if isinstance(value, dict) and "__ndarray__" in value:  # Marker belongs only to our serializer.
        return value["__ndarray__"].cpu().numpy().astype(value["dtype"], copy=True)
    if isinstance(value, dict):  # Recurse through experiment and optimizer dictionaries.
        return {key: decode_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):  # Preserve RNG tuple structure.
        return type(value)(decode_state(item) for item in value)
    return value  # Ordinary tensors remain tensors for model/optimizer restore.


def save_checkpoint(path, state):
    """Local: write then atomically replace; global: interruption cannot corrupt the last save."""
    destination = Path(path)  # Normalize caller paths without changing working directories.
    destination.parent.mkdir(parents=True, exist_ok=True)  # Create only requested output directories.
    temporary = destination.with_suffix(destination.suffix + ".tmp")  # Same filesystem enables replacement.
    torch.save(encode_state(state), temporary)  # Save only tensors and supported primitive containers.
    temporary.replace(destination)  # Publish the completed checkpoint in one filesystem operation.


def load_checkpoint(path):
    """Local: restrict deserialization; global: downloadable models need no unsafe pickle code."""
    return decode_state(torch.load(path, map_location="cpu", weights_only=True))


class PolicyCheckpoint:
    """Local: inference-only shared actor; global: evaluation and rendering share trained bots."""

    def __init__(self, path):
        """Local: load a validated actor state; global: inference never needs the central critic."""
        state = load_checkpoint(path)  # Restrict deserialization even for local model files.
        if state.get("observation_dim", OBS_DIM) != OBS_DIM:  # Reject incompatible observation versions.
            raise ValueError("Checkpoint observation dimension does not match this game")
        if tuple(state.get("schema", {}).get("action_nvecs", ACTION_NVECS)) != ACTION_NVECS:
            raise ValueError("Checkpoint control categories do not match this game")  # Prevent silent decoder mismatch.
        self.actor = ActorNetwork().eval().requires_grad_(False)  # CPU matches the simulator's arrays.
        self.actor.load_state_dict(state["actor"])  # Strict loading rejects missing/extra architecture keys.
        self.path = str(path)  # Expose provenance to reports and the renderer.

    def act(self, observation, deterministic=True):
        """Local: choose joint categories then decode; global: no scripted aiming/navigation."""
        with torch.no_grad():  # Gameplay builds no gradient graph.
            tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32).copy())
            distribution = FactorizedCategorical(self.actor(tensor))  # Local observations only.
            action = distribution.mode if deterministic else distribution.sample()  # Explicit evaluation mode.
            return decode_actions(action.numpy())  # Shared simulation decoder defines physical units.
