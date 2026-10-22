"""Local: safe model/state serialization; global: reproducible public checkpoints.

Source: https://docs.pytorch.org/docs/stable/notes/serialization.html
"""

from pathlib import Path  # Local: atomic local checkpoint paths; global: recoverability.

import numpy as np  # Local: serialize simulation arrays; global: NumPy state compatibility.
import torch  # Local: tensor-only loading; global: avoid arbitrary pickle execution.

from ministrike.learning.models import ActorNetwork, FactorizedCategorical
from ministrike.sim.constants import ACTION_NVECS, OBS_DIM, decode_actions
from ministrike.sim.specs import match_spec  # Local: validate the full match-specific interface before loading.
from ministrike.sim.constants import GameConfig  # Local: gameplay retains original checkpoint rules.
from ministrike.records import replace_artifact  # Local: share bounded Windows file-sharing recovery with JSON artifacts.


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
    replace_artifact(temporary, destination)  # Publish the complete checkpoint; brief reader locks do not discard training.


def load_checkpoint(path):
    """Local: restrict deserialization; global: downloadable models need no unsafe pickle code."""
    return decode_state(torch.load(path, map_location="cpu", weights_only=True))


class PolicyCheckpoint:
    """Local: inference-only shared actor; global: evaluation and rendering share trained bots."""

    def __init__(self, path, experiment=None):
        """Local: load a validated actor state; global: inference never needs the central critic."""
        state = load_checkpoint(path)  # Restrict deserialization even for local model files.
        self.spec = match_spec(state.get("config", {}).get("experiment"))  # Local: missing identity means original 5v5.
        if experiment is not None and self.spec.identity != experiment:
            raise ValueError("Checkpoint match does not match the requested experiment")  # Local: explicit incompatibility.
        if state.get("observation_dim", OBS_DIM) != self.spec.observation_dim:
            raise ValueError("Checkpoint observation dimension does not match this game")
        if tuple(state.get("schema", {}).get("action_nvecs", ACTION_NVECS)) != self.spec.action_nvecs:
            raise ValueError("Checkpoint control categories do not match this game")  # Prevent silent decoder mismatch.
        self.actor = ActorNetwork(self.spec.observation_dim, self.spec.action_nvecs).eval().requires_grad_(False)
        self.actor.load_state_dict(state["actor"])  # Strict loading rejects missing/extra architecture keys.
        self.path = str(path)  # Expose provenance to reports and the renderer.
        rules = state.get("environment", {}).get("sim", {}).get("config")  # Local: legacy experiments keep exact rules.
        self.rules = GameConfig(**rules) if rules else self.spec.rules()  # Local: inference never substitutes new rules.

    def act(self, observation, deterministic=True):
        """Local: choose joint categories then decode; global: no scripted aiming/navigation."""
        with torch.no_grad():  # Gameplay builds no gradient graph.
            tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32).copy())
            distribution = FactorizedCategorical(self.actor(tensor))  # Local observations only.
            action = distribution.mode if deterministic else distribution.sample()  # Explicit evaluation mode.
            return decode_actions(action.numpy(), self.spec.action_nvecs)  # Local: disabled heads stay absent.
