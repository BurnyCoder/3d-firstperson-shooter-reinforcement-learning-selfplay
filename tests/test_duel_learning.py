"""Local: duel adapter and distribution fixtures; global: verify contracts before gameplay optimizer experiments."""

import numpy as np  # Local: exact physical-control assertions.
import pytest  # Local: explicit failure contracts.
import torch  # Local: synthetic distribution checks without optimizing gameplay policies.
from tensordict import TensorDict  # Local: exercise the real selective-reset interface.

from ministrike.learning.config import TrainConfig
from ministrike.learning.environment import ArenaEnv
from ministrike.learning.models import ActorNetwork, CentralCritic, FactorizedCategorical, make_actor
from ministrike.learning.opponents import SnapshotPool
from ministrike.policies.duel import DuelRandomPolicy, DuelScriptedPolicy
from ministrike.sim.specs import match_spec
from ministrike.sim.simulation import Simulation


def test_dynamic_network_and_disabled_entropy():
    """Local: remove interaction head rather than masking sampled actions; global: PPO probability stays correct."""
    spec = match_spec("duel-fixed-v1")
    actor = ActorNetwork(spec.observation_dim, spec.action_nvecs)
    observation = torch.zeros(2, 3, 1, spec.observation_dim)
    distribution = FactorizedCategorical(actor(observation))
    assert distribution.sample().shape == (2, 3, 1, 5)
    assert distribution.entropy().shape == (2, 3, 1)
    assert CentralCritic(spec.observation_dim, 1)(observation).shape == (2, 3, 1, 1)


def test_duel_adapter_rollout_and_alternating_partial_reset():
    """Local: collect without optimizer work; global: library batch/time/player axes and frozen opponents stay intact."""
    spec = match_spec("duel-fixed-v1")
    network = ActorNetwork(spec.observation_dim, spec.action_nvecs)
    pool = SnapshotPool(network, 123)
    env = ArenaEnv(2, 123, "selfplay", pool, experiment=spec.identity)
    before = env.reset()
    assert before["agents", "observation"].shape == (2, 1, 192)
    roles = env.roles.copy()
    with torch.no_grad():
        batch = env.rollout(3, make_actor(network), auto_reset=False, break_when_any_done=False, tensordict=before)
    assert batch["agents", "action"].shape == (2, 3, 1, 5)
    other = env.sim.positions[1].copy()
    env.reset(TensorDict({"_reset": torch.tensor([[True], [False]])}, [2]))
    assert env.roles[0] == 1 - roles[0] and env.roles[1] == roles[1]
    np.testing.assert_array_equal(other, env.sim.positions[1])
    snapshot = env.snapshot()
    env.restore(snapshot)
    assert env.reset_counts.tolist() == snapshot["reset_counts"].tolist()
    env.close()


def test_evaluation_controllers_use_exact_action_grid():
    """Local: scripted/random controls respect actor quantization; global: no scripted precision advantage."""
    sim = Simulation(experiment="duel-fixed-v1")
    for policy in (DuelRandomPolicy(1), DuelScriptedPolicy()):
        actions = policy.act(sim.observe())
        assert actions.shape == (1, 2, 7)
        assert np.isin(actions[..., 2], np.array([-.24, -.08, 0, .08, .24], np.float32)).all()
        assert np.isin(actions[..., 3], np.array([-.12, -.04, 0, .04, .12], np.float32)).all()
        assert not actions[..., 6].any()


def test_no_scripted_training_or_unregistered_launch():
    """Local: fail before optimization; global: enforce pure self-play and publication prerequisites."""
    from ministrike.experiments.registration import training_preflight

    with pytest.raises(ValueError, match="self-play"):
        TrainConfig(experiment="duel-fixed-v1", condition="fixed").validate()
    with pytest.raises(ValueError, match="published"):
        training_preflight(TrainConfig(experiment="duel-fixed-v1"))
