"""Local: checkpoint milestones without gameplay or optimization; global: a duration comparison must preserve its earlier endpoint."""

import torch  # Local: compare exact checkpoint tensors and random-generator states.
import pytest  # Local: fail invalid registrations before an experiment starts.

from ministrike.learning import trainer  # Local: exercise the real checkpoint/resume orchestration.
from ministrike.learning.checkpoint import load_checkpoint
from ministrike.learning.config import TrainConfig


def test_milestone_survives_later_updates_and_resume(tmp_path, monkeypatch):
    """Local: substitute no-op collection/update fixtures; global: test persistence without an optimizer-bearing game run."""
    monkeypatch.setattr(trainer, "collect_batch", lambda *args: None)
    monkeypatch.setattr(trainer, "optimize_batch", lambda *args: {"loss": 0.0})
    config = TrainConfig(output_dir=str(tmp_path / "original"), num_envs=1, total_frames=4,
                         rollout_steps=2, snapshot_interval=2, milestone_updates=[1], torch_threads=1)
    final = trainer.train_run(config)
    milestone = tmp_path / "original/milestones/update-000001.pt"
    before, after = load_checkpoint(milestone), load_checkpoint(final["checkpoint"])
    assert before["updates"] == 1 and before["frames"] == 2
    assert after["updates"] == 2 and after["frames"] == 4
    assert torch.equal(before["torch_rng"], after["torch_rng"])  # Saving an endpoint must not sample new random actions.
    resumed = trainer.train_run(TrainConfig(output_dir=str(tmp_path / "resumed"), resume=str(milestone)))
    restored = load_checkpoint(resumed["checkpoint"])
    assert restored["updates"] == 2 and restored["config"]["milestone_updates"] == [1]
    assert all(torch.equal(value, restored["actor"][key]) for key, value in after["actor"].items())
    assert list((tmp_path / "original/milestones").iterdir()) == [milestone]  # No unregistered endpoint is silently retained.


@pytest.mark.parametrize("milestones", [[0], [-1], [2, 1], [1, 1], [1.5], [100]])
def test_invalid_milestone_registration_fails(milestones):
    """Local: reject unreachable/ambiguous endpoints; global: compare registered equal exposure."""
    with pytest.raises(ValueError, match="milestone"):
        TrainConfig(num_envs=1, total_frames=4, rollout_steps=2, milestone_updates=milestones).validate()
