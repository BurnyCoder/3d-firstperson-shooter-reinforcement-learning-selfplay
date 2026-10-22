"""Local: check nested budget enforcement without optimizer updates; global: resume cannot extend suite deadlines.

Clock source: https://docs.python.org/3/library/time.html#time.monotonic
"""

import time  # Local: create an expired monotonic deadline; global: independent of wall-clock corrections.

from ministrike.learning import trainer  # Local: patch optimizer phase; global: assert no training occurs in this unit test.
from ministrike.learning.checkpoint import load_checkpoint
from ministrike.learning.config import TrainConfig


def test_expired_suite_deadline_prevents_training_even_after_resume(tmp_path, monkeypatch):
    """Local: preserve the original run budget but honor an expired enclosing deadline."""
    def forbidden_update(*args, **kwargs):
        """Local: fail if optimization starts; global: this fixture must never perform actual training."""
        raise AssertionError("An expired absolute deadline must prevent optimizer updates")

    monkeypatch.setattr(trainer, "optimize_batch", forbidden_update)  # The test cannot accidentally train a policy.
    config = TrainConfig(output_dir=str(tmp_path), num_envs=1, total_frames=8, max_seconds=60,
                         rollout_steps=4, source_revision="unit-test")
    initial = trainer.train_run(config, deadline_monotonic=time.monotonic() - 1)  # Initialization only; no frames or optimizer steps.
    resumed = trainer.train_run(TrainConfig(output_dir=str(tmp_path), resume=initial["checkpoint"], max_seconds=999),
                                 deadline_monotonic=time.monotonic() - 1)
    assert initial["frames"] == resumed["frames"] == 0  # No game decisions were collected after the deadline.
    assert resumed["updates"] == 0 and resumed["complete"] is False  # Zero work is not a completed experiment.
    assert load_checkpoint(resumed["checkpoint"])["config"]["max_seconds"] == 60  # Resume retains the frozen run allocation.
