"""Local: baseline controller properties. Global: controls obey the learned bots' observation boundary."""

import numpy as np  # Construct exact visible/hidden observation fixtures.

from ministrike.policies.scripted import RandomPolicy, ScriptedPolicy  # Exercise public controller interfaces.
from ministrike.sim.constants import OBS_DIM  # Keep fixtures aligned with the simulation contract.


def test_random_policy_reproduces_seed():
    """Equal seeds must give identical sampled actions without global RNG dependence."""
    observation = np.zeros((10, OBS_DIM), dtype=np.float32)  # Random controls do not inspect private world state.
    np.testing.assert_array_equal(RandomPolicy(42).act(observation), RandomPolicy(42).act(observation))  # Seed isolation is observable.


def test_scripted_policy_is_finite_and_dead_players_do_nothing():
    """Absent living decisions must never produce movement or shots."""
    observation = np.zeros((2, 10, OBS_DIM), dtype=np.float32)  # Alive flags are false in every current frame.
    controls = ScriptedPolicy().act(observation)  # Test arbitrary leading batch dimensions.
    assert controls.shape == (2, 10, 7)  # Simulation controls have seven physical components.
    assert np.isfinite(controls).all() and not controls.any()  # No navigation singularity can move a dead slot.
