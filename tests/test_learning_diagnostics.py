"""Local: validate diagnostic interventions; global: an audit must not confuse broken scoring with poor learning.

Fixtures preserve the production simulator and library environment interfaces.
Source: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html
"""

import numpy as np  # Local: explicit physical controls provide deterministic ground truth.

from ministrike.experiments.diagnostics import behavior_probe
from ministrike.experiments.learning_probe import FiringRangeEnv
from ministrike.learning.models import ActorNetwork
from ministrike.learning.opponents import SnapshotPool
from ministrike.policies.scripted import ScriptedPolicy


def test_identical_scripted_policies_score_one_half_with_sides_swapped():
    """Local: use identical deterministic controllers; global: win bookkeeping must honor learner role."""
    rows = behavior_probe(ScriptedPolicy(), "combat", seed=1100009, pairs=2)
    assert len(rows) == 4 and np.mean([row["score"] for row in rows]) == .5
    assert all(row["damage"] > 0 for row in rows)  # Local: a nontrivial real firefight is being scored.


def test_firing_control_requires_a_real_shot_and_timeout_is_failure():
    """Local: contrast fire/no-fire under matched geometry; global: the learning fixture cannot award passive wins."""
    env = FiringRangeEnv(2, 91000, "fixed", SnapshotPool(ActorNetwork(), 91000))
    env.reset()  # Local: establish the real one-attacker/one-target starting state.
    env.sim.positions[:, 5, 0] = -8  # Local: aligned geometry makes the symbolic answer exact.
    controls = np.zeros((2, 10, 7), dtype=np.float32)  # Local: inactive targets and neutral player controls.
    controls[0, 0, 4] = 1  # Local: only arena zero receives the fire intervention.
    for _ in range(4):  # Local: allow the full fixed diagnostic horizon in both worlds.
        env.sim.step(controls)  # Global: use ordinary rifle hits and terminal rules.
    np.testing.assert_array_equal(env.sim.winner, [0, 1])  # Local: shooting succeeds; waiting loses.
    np.testing.assert_array_equal(env.sim.shots_fired[:, 0], [1, 0])  # Local: one actual rifle shot caused the success.
    env.close()  # Local: complete the library lifecycle.
