"""Local: analytic rotation and compatibility fixtures; global: representation changes cannot leak state or silently change old models."""

import numpy as np  # Local: independently compute known rotation expectations.
import pytest  # Local: unknown saved transformations fail explicitly.
import torch  # Local: exercise the exact CPU actor/pool/checkpoint path without gameplay training.

from ministrike.learning.observation_transform import transform_observation
from ministrike.learning.models import ActorNetwork
from ministrike.learning.opponents import SnapshotPool
from ministrike.learning.checkpoint import PolicyCheckpoint, save_checkpoint
from ministrike.sim.simulation import Simulation


def test_eye_coordinates_use_each_historical_pose_and_preserve_masks():
    """Local: two analytically known camera poses; global: history is transformed without adding hidden fields."""
    frames = torch.zeros(2, 4, 48)
    frames[..., 4] = frames[..., 6] = 1  # Local: yaw/pitch zero in three of four frames.
    frames[:, -1, 3:7] = torch.tensor([1., 0., .5, np.sqrt(.75)])
    frames[..., 42:45] = torch.tensor([1., 2., 3.])
    frames[..., 45] = 1  # Local: visible opponent row; own hidden row remains zero.
    original = frames.clone()
    actual = transform_observation(frames.flatten(-2), "egocentric-v1").reshape_as(frames)
    assert torch.equal(frames, original)  # Local: cached simulator observations must remain immutable.
    torch.testing.assert_close(actual[:, :-1], original[:, :-1])
    expected = torch.tensor([-3., 2 * np.sqrt(.75) - .5, np.sqrt(.75) + 1], dtype=torch.float32)
    torch.testing.assert_close(actual[:, -1, 42:45], expected.expand(2, 3))
    torch.testing.assert_close(actual[..., 42:45].norm(dim=-1), original[..., 42:45].norm(dim=-1))
    assert torch.equal(actual[..., :36], original[..., :36])
    assert torch.equal(actual[..., 36:42], original[..., 36:42])
    assert torch.equal(actual[..., 45:], original[..., 45:])


def test_identity_and_invalid_transform_contract():
    """Local: original actor input is returned exactly; global: legacy behavior is not reinterpreted."""
    observation = torch.randn(2, 1, 192)
    assert transform_observation(observation, "identity") is observation
    with pytest.raises(ValueError, match="transform"):
        transform_observation(observation, "unknown")


def test_matching_initial_weights_and_frozen_opponent_preprocessing(tmp_path):
    """Local: same tensors, different explicit representation; global: all inference/resume paths retain the variant."""
    torch.manual_seed(5201)
    original = ActorNetwork(192, (9, 5, 5, 2, 2))
    torch.manual_seed(5201)
    variant = ActorNetwork(192, (9, 5, 5, 2, 2), observation_transform="egocentric-v1")
    assert all(torch.equal(value, variant.state_dict()[key]) for key, value in original.state_dict().items())
    observation = Simulation(experiment="duel-fixed-v1").observe()
    expected = original(transform_observation(torch.from_numpy(observation), "egocentric-v1"))
    torch.testing.assert_close(variant(torch.from_numpy(observation)), expected)
    pool = SnapshotPool(variant, 9001)
    pool.act(observation, np.array([0]))  # Local: build a frozen cache without advancing a game or optimizer.
    assert pool.cache[0].observation_transform == "egocentric-v1"
    torch.testing.assert_close(pool.cache[0](torch.from_numpy(observation)), expected)
    saved = pool.state_dict()
    restored = SnapshotPool(variant, 9002)
    restored.load_state_dict(saved)
    assert restored.observation_transform == "egocentric-v1"
    with pytest.raises(ValueError, match="transform"):
        SnapshotPool(original, 9003).load_state_dict(saved)
    path = tmp_path / "actor.pt"
    state = {"actor": variant.state_dict(), "observation_dim": 192,
             "config": {"experiment": "duel-fixed-v1", "observation_transform": "egocentric-v1"},
             "schema": {"action_nvecs": (9, 5, 5, 2, 2), "actor_observation_transform": "egocentric-v1"}}
    save_checkpoint(path, state)
    loaded = PolicyCheckpoint(path, "duel-fixed-v1")
    torch.testing.assert_close(loaded.actor(torch.from_numpy(observation)), expected)
    state["schema"]["actor_observation_transform"] = "identity"
    save_checkpoint(path, state)
    with pytest.raises(ValueError, match="transform"):
        PolicyCheckpoint(path, "duel-fixed-v1")


def test_resume_comparison_rejects_numerical_or_dtype_drift():
    """Local: nested checkpoint differences cannot pass the audit; global: real resume acceptance requires exact numerical state."""
    from ministrike.experiments.resume_verification import state_equal

    before = {"tensor": torch.tensor([1.]), "array": np.array([2.]), "rng": (3, [4])}
    after = {"tensor": before["tensor"].clone(), "array": before["array"].copy(), "rng": (3, [4])}
    assert state_equal(before, after)
    after["tensor"] += 1
    assert not state_equal(before, after)
    assert not state_equal(torch.tensor([1.]), torch.tensor([1]))
    assert not state_equal(np.array([1.]), np.array([1]))
