"""Local: synthetic controls for action-mode interventions; global: never confuse both-controller changes with learner-only effects."""

from dataclasses import replace  # Local: shorten a rule fixture without changing registered gameplay.
from datetime import datetime, timezone, timedelta  # Local: expired deadlines must stop before an evaluated transition.
from types import SimpleNamespace  # Local: checkpoint stubs carry no trained weights or optimizer.

import numpy as np  # Local: exact legal controls and state comparisons.
import pytest  # Local: cover the four prespecified mode combinations.

from ministrike.experiments import duel_evaluation as evaluation
from ministrike.experiments.duel_trace import trace_step
from ministrike.experiments.mode_analysis import compare_modes
from ministrike.sim.simulation import Simulation
from ministrike.sim.specs import match_spec


@pytest.mark.parametrize("learner,opponent", [(False, False), (False, True), (True, False), (True, True)])
def test_modes_are_independent_and_randomness_determines_unique_matches(tmp_path, monkeypatch, learner, opponent):
    """Local: record each actual controller call; global: hold opponent mode fixed when intervening on the learner."""
    rules = replace(match_spec("duel-fixed-v1").rules(), round_seconds=.05)
    monkeypatch.setattr(evaluation, "PolicyCheckpoint", lambda path, _: SimpleNamespace(label=path, rules=rules))
    monkeypatch.setattr(evaluation, "file_hash", str)
    calls = []  # Local: this fixture tests routing and match counts, not policy quality.
    def actions(policy, observations, generators, deterministic):
        """Local: record selection mode and return zero controls; global: no gameplay optimizer or training data."""
        calls.append((policy.label, deterministic))
        return np.zeros((*observations.shape[:-1], 7), np.float32)
    monkeypatch.setattr(evaluation, "policy_actions", actions)
    result = evaluation.evaluate_duel("learner", "opponent", tmp_path / "eval.json", pairs=2,
                                      stochastic=learner, initial_stochastic=opponent, baselines=("initial",), trace_first_seed=True)
    assert set(calls) == {("learner", not learner), ("opponent", not opponent)}
    assert len(result["rounds"]) == (4 if learner or opponent else 2)
    assert result["summary"]["score"] == .5 and len(result["traces"]) == 2
    expired = evaluation.evaluate_duel("learner", "opponent", tmp_path / "expired.json", baselines=("initial",),
                                       deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert expired["status"] == "budget_exhausted" and not expired["rounds"]


def test_trace_is_observational_and_masks_aim_error_when_hidden():
    """Local: tracing cannot change state or RNG; global: evidence instrumentation cannot alter the experiment."""
    sim = Simulation(experiment="duel-fixed-v1")
    before = sim.state_dict()
    record = trace_step(sim, sim.observe(), np.zeros((1, 2, 7), np.float32), 0)
    assert record["players"][0]["visible"] and record["players"][0]["aim_error_degrees"] > 0
    for key in sim._STATE_ARRAYS:
        np.testing.assert_array_equal(getattr(sim, key), before["arrays"][key])
    assert sim.rng.bit_generator.state == before["rng"]
    sim.yaw[0, 0] = np.pi
    sim.observe()  # Local: refresh the current frame after this explicit synthetic visibility intervention.
    hidden = trace_step(sim, sim.observe(), np.zeros((1, 2, 7), np.float32), 0)
    assert not hidden["players"][0]["visible"] and hidden["players"][0]["aim_error_degrees"] is None


def test_mode_effect_keeps_seed_and_side_pairing_and_rejects_duplicate_rows():
    """Local: opposing end effects cancel exactly; global: match repetition does not create independent model evidence."""
    def cell(mode, scores):
        """Local: build hand-computable paired trajectories; global: deterministic cells have only one world per end."""
        seeds = [3_000_000] if mode == "DD" else [3_000_000, 3_000_001]
        return {"identity": {"seed_base": 3_000_000, "pairs": 2},
                "rounds": [{"side": side, "seed": seed, "score": scores[side], "baseline": "initial"}
                           for seed in seeds for side in (0, 1)]}
    cells = {mode: cell(mode, [.5, .5] if mode[0] == "D" else [1, .5]) for mode in ("DD", "DS", "SD", "SS")}
    result = compare_modes({"R0": cells})
    assert result["primary"]["delta"] == .25 and result["primary"]["ci95"] == [.25, .25]
    cells["SD"]["rounds"].append(cells["SD"]["rounds"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        compare_modes({"R0": cells})
