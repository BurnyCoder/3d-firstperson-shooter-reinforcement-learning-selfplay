"""Local: frozen experiment bookkeeping regressions. Global: incomplete compute must never become completed evidence."""

from datetime import UTC, datetime, timedelta  # Local: give fixtures a valid shared deadline without running training.

from ministrike.experiments.suite import RUN_ORDER, run_suite  # Local: exercise the actual resume/completion wrapper.
from ministrike.records import write_json  # Local: use the production atomic evidence format.


def test_suite_does_not_mark_unequal_frame_runs_complete(tmp_path):
    """Local: one timed-out run invalidates full completion. Global: six artifacts do not imply six equal-budget experiments."""
    runs = {}  # Local: construct recorded results without optimizing any policy.
    for index, (condition, seed) in enumerate(RUN_ORDER):  # Local: include every pre-specified experimental identity.
        name = f"{condition}_seed{seed}"  # Local: production directory naming is part of the resume contract.
        runs[name] = {"complete": index != 0, "frames": 128 if index else 64}  # Local: one run missed its frozen target.
        write_json(tmp_path / name / "result.json", runs[name])  # Local: resume must retain the time-limited result.
        write_json(tmp_path / name / "evaluation.json", {"status": "complete"})  # Local: evaluation itself finished.
        write_json(tmp_path / name / "initial_evaluation.json", {"status": "complete"})  # Local: both baseline endpoints exist.
    manifest = {"output_dir": str(tmp_path), "deadline_utc": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                "num_envs": 1, "evaluation_pairs": 1, "total_frames": 128, "per_run_seconds": 30,
                "device": "cpu", "runs": runs, "test_seed_base": 2_000_000}  # Local: bypass calibration using a frozen protocol.
    manifest_path = tmp_path / "suite.json"  # Local: this is the explicit resume entry point.
    write_json(manifest_path, manifest)  # Local: persist the pre-existing suite before invoking production code.
    result = run_suite(tmp_path, resume=manifest_path)  # Local: no learning or evaluation should be repeated.
    assert result["status"] == "budget_exhausted"  # Local: completed tournaments cannot conceal incomplete training.
    assert result["runs"]["selfplay_seed111"]["frames"] == 64  # Local: the partial experimental outcome remains unchanged.
