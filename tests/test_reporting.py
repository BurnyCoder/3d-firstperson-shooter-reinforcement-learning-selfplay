"""Local: verify report inclusion and missingness; global: prevent fabricated or selected scientific claims.

Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
"""

import json  # Local: measured-artifact fixtures; global: reproduce report discovery contracts.

from ministrike.experiments.reporting import discover_runs, paired_training_comparison


def _record(condition, seed, score):
    """Local: construct a synthetic summary fixture; global: no fake experiment files are published."""
    return {"primary": True, "config": {}, "result": {"condition": condition, "seed": seed,
            "frames": 1000, "complete": True}, "evaluation": {"status": "complete", "summary": {"mean_score": score}}}


def test_comparison_requires_every_registered_seed():
    """Local: reject convenient partial subsets; global: no post-hoc favorable-seed selection."""
    records = [_record("selfplay", 111, 1.0), _record("fixed", 111, 0.0)]  # A favorable single pair is insufficient.
    comparison = paired_training_comparison(records)  # Apply the production inclusion rule.
    assert comparison["status"] == "pending" and comparison["mean_difference"] is None
    assert comparison["ci95"] is None and len(comparison["pairs"]) == 1  # Retain evidence without claiming completion.


def test_three_seed_interval_uses_training_differences():
    """Local: use all independent model pairs; global: distinguish retraining uncertainty from matches."""
    records = []  # Each seed contributes exactly one algorithm difference.
    for seed, score in ((111, 0.4), (222, 0.5), (333, 0.6)):
        records.extend([_record("selfplay", seed, score), _record("fixed", seed, 0.3)])
    comparison = paired_training_comparison(records)  # Pair by seed, not by individual test round.
    assert comparison["status"] == "complete"  # All three registered pairs are present.
    assert abs(comparison["mean_difference"] - 0.2) < 1e-12  # Equal weighting across trained models.
    assert comparison["ci95"][0] < 0 < comparison["ci95"][1]  # Three-seed uncertainty remains visibly wide.


def test_unequal_budget_and_duplicate_runs_are_not_selected():
    """Local: preserve mismatches as exclusions; global: the report cannot cherry-pick a matching result."""
    records = [_record(condition, seed, 0.5) for seed in (111, 222, 333) for condition in ("selfplay", "fixed")]
    records[0]["result"]["frames"] = 2000  # Unequal compute breaks the declared comparison.
    records.append(_record("fixed", 222, 0.1))  # A duplicate with an appealing score is deliberately ambiguous.
    comparison = paired_training_comparison(records)  # Never choose the result maximizing the contrast.
    assert comparison["status"] == "pending" and len(comparison["pairs"]) == 1
    assert len(comparison["excluded"]) == 2  # Both defects remain visible to the report reader.


def test_discovery_preserves_missing_evaluation_and_labels_calibration(tmp_path):
    """Local: read actual JSON layout; global: absence is pending and calibration is not a primary run."""
    path = tmp_path / "calibration" / "selfplay_seed111"  # Naming alone must not admit a calibration run.
    path.mkdir(parents=True)  # Pytest owns all temporary fixture artifacts.
    (path / "result.json").write_text(json.dumps({"condition": "selfplay", "seed": 111, "frames": 32}), encoding="utf-8")
    records = discover_runs(tmp_path)  # Read the same paths used by the production report generator.
    assert len(records) == 1 and records[0]["evaluation"] is None  # No fabricated zero score.
    assert records[0]["primary"] is False  # Calibration evidence is visible but excluded from the endpoint.
