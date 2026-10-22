"""Local: registered duel contracts; global: prevent a second, inconsistent game engine.

Ground truth: docs/duel-protocol.md (approved requirements, not empirical claims).
"""

import numpy as np  # Local: explicit simultaneous controls and exact state comparisons.
import pytest  # Local: parameterized rule/reward fixtures, never training experiments.

from ministrike.sim.simulation import Simulation


def duel(count=1, reward="R0"):
    """Local: construct the approved match; global: exercise the same API as deployment."""
    return Simulation(count, experiment="duel-fixed-v1", reward=reward)


def test_fixed_geometry_and_observation_contract():
    """Local: fixed means no spawn RNG; global: distinguish Stage A from random-spawn B."""
    sim = duel(2)
    first = sim.reset(seed=7)
    assert first.shape == (2, 2, 192)
    np.testing.assert_array_equal(first, sim.reset(seed=99))
    assert sim.positions[0, 1, 2] - sim.positions[0, 0, 2] == 6
    assert sim.walls.shape == (4, 6)
    assert sim.config.round_seconds == 20
    assert sim.spec.action_nvecs == (9, 5, 5, 2, 2)


@pytest.mark.parametrize("reward,alpha", [("R0", .25), ("R1", .5)])
def test_effective_damage_and_elimination(reward, alpha):
    """Local: cap overkill; global: reward accounting cannot exploit low-health victims."""
    sim = duel(reward=reward)
    sim.health[0, 1] = 10
    action = np.zeros((1, 2, 7), np.float32)
    action[0, 0, 4] = 1
    result = sim.step(action)
    assert result.terminated[0] and result.winner[0] == 0
    np.testing.assert_allclose(result.rewards, [[1 + alpha * .1, -1 - alpha * .1]])
    assert sim.damage_dealt[0, 0] == 10


def test_simultaneous_lethal_and_timeout_draw():
    """Local: resolve both shots before outcomes; global: player ordering cannot decide wins."""
    sim = duel()
    sim.health[:] = 25
    actions = np.zeros((1, 2, 7), np.float32)
    actions[..., 4] = 1
    result = sim.step(actions)
    assert result.terminated[0] and result.winner[0] == -1
    np.testing.assert_array_equal(result.rewards, 0)
    sim.reset()
    actions[:] = 0
    actions[..., 6] = 1
    for _ in range(400):
        result = sim.step(actions)
    assert result.terminated[0] and result.winner[0] == -1
    assert not sim.bomb_planted.any()


def test_partial_reset_and_legacy_incompatibility():
    """Local: selective reset retains the other arena; global: resume validates match identity."""
    sim = duel(2)
    sim.step(np.zeros((2, 2, 7), np.float32))
    before = sim.history[1].copy()
    sim.reset(mask=np.array([True, False]))
    assert sim.tick.tolist() == [0, 1]
    np.testing.assert_array_equal(sim.history[1], before)
    with pytest.raises(ValueError, match="match|rules|shape"):
        Simulation(2).load_state_dict(sim.state_dict())


def test_potential_telescopes_at_terminal():
    """Local: discounted sum equals -initial potential; global: no terminal aim bonus leaks.

    Identity: Ng et al. (1999), https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
    """
    from ministrike.sim.rewards import aim_potential

    sim = duel(reward="R2")
    initial = aim_potential(sim).copy()
    discount, total = 1.0, np.zeros((1, 2))
    for _ in range(400):
        sim.step(np.zeros((1, 2, 7), np.float32))
        total += discount * sim.reward_components["potential"]
        discount *= .999
    np.testing.assert_allclose(total, -initial, atol=1e-6)
