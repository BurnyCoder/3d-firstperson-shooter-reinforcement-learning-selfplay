"""Local: synthetic analysis controls; global: no actual test-set scores influence the analysis implementation."""

import pytest  # Local: enforce pairing integrity and numeric expectations.

from ministrike.experiments.duel_analysis import paired_match_interval, paired_power, model_comparison


def evaluation(score):
    """Local: hand-constructed identical paired cells; global: deterministic repeats create no variance."""
    return {"rounds": [{"baseline": name, "side": side, "seed": 4_000_000, "score": score}
                       for name in ("random", "initial", "scripted") for side in (0, 1)],
            "summary": {"score": score, "elimination_fraction": 1,
                        "baselines": {name: {"score": score} for name in ("random", "initial", "scripted")}}}


def test_paired_analysis_and_three_model_gate():
    """Local: known differences retain exact arithmetic; global: missing seeds cannot pass confirmation."""
    before, after = evaluation(.5), evaluation(1)
    assert paired_match_interval(before, after)["ci95"] == [.5, .5]
    rows = [{"seed": seed, "complete": True, "initial": before, "final": after} for seed in (5101, 5102, 5103)]
    assert model_comparison(rows)["mean_improvement"] == .5
    assert model_comparison(rows)["acceptance"]
    assert model_comparison(rows[:-1])["status"] == "incomplete"
    after["rounds"].append(after["rounds"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_match_interval(before, after)


def test_power_sensitivity_is_not_a_claim_of_adequacy():
    """Local: increasing assumed noise reduces power; global: use precisely three independent differences."""
    rows = paired_power()
    assert [row["assumed_sd"] for row in rows] == [.05, .10, .20]
    assert rows[0]["power"] > rows[1]["power"] > rows[2]["power"]
    assert all(row["models"] == 3 and 0 < row["power"] < 1 for row in rows)
