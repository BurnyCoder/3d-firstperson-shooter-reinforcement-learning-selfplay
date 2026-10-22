"""Local: verify report inclusion and missingness; global: prevent fabricated or selected scientific claims.

Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
"""

import json  # Local: measured-artifact fixtures; global: reproduce report discovery contracts.

from ministrike.experiments.reporting import _paper_sections, _preserve_suite, _suite_appendix, discover_runs, paired_training_comparison


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


def test_discovery_excludes_generated_publication_copies(tmp_path):
    """Local: upload staging is not new evidence. Global: publishing must not duplicate training outcomes in reports."""
    artifact = json.dumps({"condition": "selfplay", "seed": 111, "frames": 32, "complete": False})
    for relative in ("selfplay_seed111", "publication/selfplay_seed111"):  # Local: publisher stages a copy of this original run.
        path = tmp_path / relative  # Local: both fixtures remain inside the explicitly requested evidence tree.
        path.mkdir(parents=True)  # Local: test storage contains no model files or real published artifacts.
        (path / "result.json").write_text(artifact, encoding="utf-8")  # Local: identical measured evidence appears twice on disk.
    records = discover_runs(tmp_path)  # Local: production discovery must identify only the original invocation.
    assert len(records) == 1 and records[0]["name"] == "selfplay_seed111"  # Local: no false duplicate seeds or experiments.
    assert records[0]["result"]["complete"] is False  # Local: excluding upload copies never hides the original incomplete result.


def test_suite_appendix_uses_preserved_manifest_values_and_keeps_failures(tmp_path):
    """Local: report actual protocol settings. Global: compute limits and power assumptions remain auditable."""
    source = tmp_path / "input"  # Local: represent the explicitly selected suite directory.
    destination = tmp_path / "reports"  # Local: keep generated test data separate from source evidence.
    source.mkdir()  # Local: no report/PDF rendering is needed for this metadata regression.
    suite = {"status": "failed", "started_utc": "2026-10-22T20:00:00+00:00", "deadline_utc": "2026-10-23T08:00:00+00:00",
             "max_hours": 12, "device": "cuda", "num_envs": 16, "total_frames": 123456, "per_run_seconds": 5940,
             "calibration": [{"condition": "fixed", "device": "cuda", "frames": 8192, "frames_per_second": 287.125}],
             "power_sensitivity": [{"assumed_run_sd": .10, "effect": .10, "alpha": .05, "target_power": .80,
                                    "seeds_per_condition": 17, "achieved_power": .807}], "error": "recorded fixture interruption"}
    raw = (json.dumps(suite, indent=3) + "\n").encode("utf-8")  # Local: nondefault spacing verifies byte preservation.
    (source / "suite.json").write_bytes(raw)  # Local: establish the only protocol used for this report.
    restored = _preserve_suite(source, destination)  # Local: copy exact bytes and parse that same snapshot.
    assert (destination / "data" / "suite.json").read_bytes() == raw  # Local: all original records, including failures, survive unchanged.
    text = _suite_appendix(restored)  # Local: compose prose only; parent agent controls subsequent PDF generation.
    for expected in ("2026-10-23T08:00:00+00:00", "cuda", "123456", "5940.0", "287.12", "0.807", "17", "recorded fixture interruption"):
        assert expected in text  # Local: appendix uses recorded values rather than defaults or invented measurements.
    assert "independent two-group" in text and "final three-seed paired analysis" in text  # Local: distinguish power-model assumptions.
    assert _preserve_suite(destination / "data", destination) == suite  # Local: already copied source/destination identity is safe.


def test_missing_suite_metadata_is_pending_and_abstract_grammar_is_count_independent():
    """Local: missing protocol is explicit. Global: standalone smoke evidence cannot masquerade as the planned suite."""
    comparison = paired_training_comparison([])  # Local: no fabricated final comparison.
    sections = dict(_paper_sections([], comparison))  # Local: pure text composition does not generate a report artifact.
    assert "Recorded training invocations: 0" in sections["Abstract"]  # Local: count-independent phrase avoids singular/plural mistakes.
    assert "power sensitivity are pending" in sections["Reproducibility appendix"]  # Local: missing manifest remains unknown.
    assert "Noncentral-t power calculation" in sections["Sources"]  # Local: the appendix's method keeps its primary reference.
