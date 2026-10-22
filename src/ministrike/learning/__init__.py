"""Local: small learning API; global: callers need no internal TorchRL implementation details."""

from ministrike.learning.checkpoint import PolicyCheckpoint  # Public inference interface for play/evaluation.
from ministrike.learning.config import TrainConfig  # Public typed configuration for run orchestration.
from ministrike.learning.trainer import calibrate, train_run  # Public benchmark and training phases.

__all__ = ["PolicyCheckpoint", "TrainConfig", "calibrate", "train_run"]  # Stable documented package exports.
