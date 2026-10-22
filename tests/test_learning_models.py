"""Local: test policy probability axes; global: guard the MAPPO implementation.

Source: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
"""

import torch  # Local: synthetic inputs; global: differentiable policy checks.

from ministrike.learning.models import ActorNetwork, CentralCritic, FactorizedCategorical
from ministrike.learning.opponents import SnapshotPool
from ministrike.sim.constants import ACTION_NVECS, OBS_DIM


def test_factorized_distribution_keeps_agents_separate():
    """Local: sum control heads only; global: one PPO ratio per player."""
    logits = ActorNetwork()(torch.zeros(2, 5, OBS_DIM))  # Produce player-specific heads.
    distribution = FactorizedCategorical(logits)  # Interpret the final two axes as controls.
    action = distribution.sample()  # Draw one joint action for each player.
    assert action.shape == (2, 5, 6)  # Six control heads must survive sampling.
    assert distribution.log_prob(action).shape == (2, 5)  # Agent axes must not be summed.
    assert distribution.entropy().shape == (2, 5)  # Entropy follows the same convention.
    for head, size in enumerate(ACTION_NVECS):  # Every sampled category is legal.
        assert torch.all(action[..., head] < size)


def test_central_critic_preserves_player_slots():
    """Local: verify critic dimensions; global: terminal rewards reach dead slots."""
    observation = torch.randn(2, 7, 5, OBS_DIM)  # Batch, time, player, observation.
    assert CentralCritic()(observation).shape == (2, 7, 5, 1)  # One value per retained player.


def test_snapshot_is_immutable_and_initial_is_retained():
    """Local: mutate learner after snapshots; global: avoid moving opponents."""
    actor = ActorNetwork()  # Build a fresh learner.
    pool = SnapshotPool(actor, seed=7, capacity=8)  # Preserve its initial policy.
    original = {key: value.clone() for key, value in pool.states[0].items()}  # Capture evidence.
    for index in range(12):  # Exceed historical capacity deliberately.
        with torch.no_grad():  # Parameter edits are test fixtures, not optimizer training.
            next(actor.parameters()).add_(0.01)
        pool.add(actor, index + 1)  # Every entry must own its weights.
        pool.prune()  # Discard snapshots after all rounds using them have ended.
    assert len(pool.states) <= 10  # Initial plus latest and eight historical entries.
    assert all(torch.equal(pool.states[0][key], value) for key, value in original.items())
    assert set(pool.choose(200)).issubset(pool.states)  # Sampling must use retained identities.
