"""Local: a registered tiny real-training resume control; global: diagnostic weights never deploy.

This uses the production collector/optimizer/checkpoint APIs, not mocked updates.
Serialization contract: https://docs.pytorch.org/docs/stable/notes/serialization.html .
"""

from pathlib import Path  # Local: explicit registered checkpoint and report paths.
from shutil import copy2  # Local: preserve the original initial baseline for the resumed artifact pair.

import numpy as np  # Local: compare native simulator and RNG arrays exactly.
import torch  # Local: compare optimizer/model tensors without approximate tolerances.

from ministrike.experiments.registration import published_preflight, read_json  # Local: registration precedes optimization.
from ministrike.learning.checkpoint import load_checkpoint  # Local: weights-only safe loading.
from ministrike.learning.config import TrainConfig  # Local: identical production configuration contract.
from ministrike.learning.trainer import train_run  # Local: exercise the actual collection/GAE/PPO/Adam/resume phases.
from ministrike.records import timestamp, write_json  # Local: durable success or failure, including identities.


def state_equal(first, second):
    """Local: recursively compare exact saved numerical state; global: elapsed time and file metadata are excluded by the caller."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and first.dtype == second.dtype and np.array_equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(state_equal(value, second[key]) for key, value in first.items())
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(state_equal(a, b) for a, b in zip(first, second, strict=True))
    return first == second  # Local: scalar counters and strings have exact equality semantics.


def verify_real_resume(config_path):
    """Local: train two updates, branch at one, finish and compare; global: no artificial interruption of a healthy long worker."""
    config = read_json(config_path)
    preflight = published_preflight(config_path, "A-DIAG-RESUME-1")
    destination = Path(config["report"])
    if destination.exists():
        saved = read_json(destination)
        if saved.get("passed") and saved["preflight"]["source_revision"] == preflight["source_revision"]:
            return saved  # Local: never repeat a completed verification for the same published implementation.
        raise ValueError("Preserve and inspect the existing resume-control result before another attempt")
    rows = config["runs"]
    train_run(TrainConfig(**rows[0], run_config=str(config_path)))
    milestone = Path(rows[0]["output_dir"]) / "milestones/update-000001.pt"
    Path(rows[1]["output_dir"]).mkdir(parents=True, exist_ok=True)
    copy2(Path(rows[0]["output_dir"]) / "initial.pt", Path(rows[1]["output_dir"]) / "initial.pt")
    train_run(TrainConfig(**rows[1], run_config=str(config_path), resume=str(milestone)))
    before, after = [load_checkpoint(Path(row["output_dir"]) / "model.pt") for row in rows]
    fields = ("actor", "critic", "optimizer", "pool", "environment", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "frames", "updates", "schema")
    matches = {key: state_equal(before[key], after[key]) for key in fields}
    result = {"created_utc": timestamp(), "preflight": preflight, "matches": matches, "passed": all(matches.values()),
              "baseline_arena_steps": before["frames"], "resumed_from_arena_steps": 256,
              "additional_resumed_arena_steps": after["frames"] - 256, "deployment_eligible": False}
    write_json(destination, result)  # Local: failures remain available instead of being replaced by a favorable retry.
    if not result["passed"]:
        raise ValueError("Real optimizer resume diverged; inspect the preserved checkpoints before long training")
    return result
