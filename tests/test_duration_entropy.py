"""Local: synthetic diagnostic selection; global: a performance hypothesis is separate from recipe eligibility."""

import pytest  # Local: incomplete evidence must fail closed.

from ministrike.experiments.duration_entropy import compare_duration_entropy


def test_comparison_does_not_promote_a_losing_entropy_variant():
    """Local: known paired scores; global: every registered contrast remains visible even when negative."""
    def endpoint(score):
        """Local: hand-constructed two-end records; global: no simulated gameplay or optimizer runs in this test."""
        return {"status": "complete", "rounds": [
            {"baseline": baseline, "side": side, "seed": 3_000_000, "score": score}
            for baseline in ("random", "initial", "scripted") for side in (0, 1)],
            "summary": {"score": score, "elimination_fraction": 1.0 if score == 1 else 0.0,
                        "baselines": {baseline: {"score": score} for baseline in ("random", "initial", "scripted")}}}
    control = {"initial": endpoint(.5), "early": endpoint(.5), "final": endpoint(1.0)}
    lower = {"initial": endpoint(.5), "early": endpoint(.5), "final": endpoint(.5)}
    result = compare_duration_entropy({"control": control, "low-entropy": lower})
    assert result["duration"]["delta"] == .5 and result["duration"]["threshold_met"]
    assert result["entropy"]["delta"] == -.5 and not result["entropy"]["threshold_met"]
    assert result["eligible_recipe"] == "control" and result["deployment_eligible"] is False
    assert not result["recipes"]["low-entropy"]["screen_passed"]
    lower["final"]["status"] = "budget_exhausted"
    with pytest.raises(ValueError, match="incomplete"):
        compare_duration_entropy({"control": control, "low-entropy": lower})
