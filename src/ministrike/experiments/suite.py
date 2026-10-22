"""Local: bounded calibration/train/evaluate phases. Global: freeze a fair six-run comparison before learning.

Method sources: https://arxiv.org/abs/1709.06560 and https://arxiv.org/abs/1806.08295 .
The suite compares a small number of independent training seeds; it never trains until significance.
"""

import json  # Resume the exact manifest instead of reselecting favorable settings.
import logging  # Keep complete phase progress in the application's timestamped log.
import time  # Use monotonic time for in-process deadlines.
from datetime import UTC, datetime, timedelta  # Persist an absolute deadline across crashes/restarts.
from pathlib import Path  # All run directories remain under the requested suite folder.

import torch  # Discover supported optimization devices without assuming GPU speedups.

from ministrike.experiments.analysis import power_sensitivity  # Pre-training conditional sample-size calculation.
from ministrike.experiments.evaluation import evaluate_policy  # Use the exact public evaluation routine.
from ministrike.learning import PolicyCheckpoint, TrainConfig, calibrate, train_run  # Existing optimization implementation.
from ministrike.records import source_revision, write_json  # Atomic manifests preserve reproducible intermediate state.

LOGGER = logging.getLogger(__name__)  # Root CLI installs terminal and file handlers.
RUN_ORDER = [("selfplay", 111), ("fixed", 111), ("fixed", 222), ("selfplay", 222), ("selfplay", 333), ("fixed", 333)]  # Counterbalance sequential condition order.


def remaining_seconds(manifest: dict) -> float:
    """Local: calculate the original deadline's remaining allowance. Global: resume never grants more compute."""
    return max(0.0, (datetime.fromisoformat(manifest["deadline_utc"]) - datetime.now(UTC)).total_seconds())  # Include time since a crash or machine suspension.


def run_suite(output: str | Path, max_hours: float = 12, device: str = "auto", num_envs: int = 16, pairs: int = 128, resume: Path | None = None) -> dict:
    """Local: orchestrate independent runs. Global: freeze comparable budgets and publish all outcomes."""
    directory = Path(output).resolve()  # Save absolute artifact paths for later desktop launcher use.
    directory.mkdir(parents=True, exist_ok=True)  # Keep all calibration/training/evaluation evidence together.
    manifest_path = Path(resume) if resume else directory / "suite.json"  # Explicit resume targets an existing frozen protocol.
    if resume:  # Reload earlier decisions; CLI defaults cannot silently change an experiment.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # Includes original deadline and completed phases.
        directory = Path(manifest["output_dir"])  # Preserve the original suite artifact directory.
        num_envs, pairs = manifest["num_envs"], manifest["evaluation_pairs"]  # Prevent sample/batch changes on resume.
    else:
        if manifest_path.exists():  # Refuse to overwrite evidence from a previous invocation.
            raise FileExistsError(f"Suite already exists; resume with --resume {manifest_path}")  # Make experiment identity explicit.
        if not 0 < max_hours <= 12 or num_envs < 1 or pairs < 1:  # The agreed global cap is enforced in code.
            raise ValueError("Suite requires 0 < max_hours <= 12 and positive environments/pairs")  # No accidental unbounded training.
        now = datetime.now(UTC)  # Record when the compute budget began.
        manifest = {"version": 1, "status": "calibration", "source_revision": source_revision(), "output_dir": str(directory),
                    "started_utc": now.isoformat(), "deadline_utc": (now + timedelta(hours=max_hours)).isoformat(),
                    "max_hours": max_hours, "num_envs": num_envs, "evaluation_pairs": pairs,
                    "hypothesis": "Historical self-play increases fixed-benchmark mean round score by at least 0.10 over fixed-opponent training.",
                    "power_sensitivity": power_sensitivity(), "calibration": [], "runs": {},
                    "training_seeds": [111, 222, 333], "validation_seed_base": 1_000_000, "test_seed_base": 2_000_000,
                    "evaluation_baselines": ["random", "rush", "guard"], "run_order": RUN_ORDER}
        write_json(manifest_path, manifest)  # Persist the endpoint and analysis plan before the first calibration update.
    try:  # Failure updates the durable suite status before propagating to automation.
        if "total_frames" not in manifest:  # Calibration is performed once, never repeated to search for good scores.
            devices = [device] if device != "auto" else ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])  # Benchmark only available hardware.
            for candidate in devices:  # Device choice uses throughput exclusively, never evaluation reward.
                for condition in ("selfplay", "fixed"):  # The slower condition bounds the equal frame allowance.
                    name = f"calibration_{candidate}_{condition}"  # Calibration is disjoint from the three experimental seeds.
                    path = directory / "calibration" / name  # Keep pilot learning separate from primary results.
                    result_path = path / "result.json"  # Completed calibration can survive process interruption.
                    if result_path.exists():  # Reuse its measured timing instead of performing more pilot training.
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    else:
                        result = calibrate(TrainConfig(output_dir=str(path), condition=condition, seed=8001,
                                                      num_envs=num_envs, device=candidate, max_seconds=min(180, remaining_seconds(manifest))))  # Includes actual PPO updates/checkpoint writes.
                    result["device"] = candidate  # Record device with the measured full-pipeline throughput.
                    manifest["calibration"] = [row for row in manifest["calibration"] if not (row["device"] == candidate and row["condition"] == condition)] + [result]  # Idempotent resume bookkeeping.
                    write_json(manifest_path, manifest)  # Calibration evidence is durable after each probe.
            rates = {candidate: min(row["frames_per_second"] for row in manifest["calibration"] if row["device"] == candidate) for candidate in devices}  # Protect the slower of the two conditions.
            selected = max(rates, key=rates.get)  # Deterministic throughput-only optimizer-device selection.
            per_run = min(manifest["max_hours"] * 3600 * 5 / 36, remaining_seconds(manifest) * 5 / 36)  # Six runs share at most five-sixths of the global budget.
            quantum = num_envs * 128  # Complete rollout batches give all conditions identical transition counts.
            frames = max(quantum, int(rates[selected] * per_run * 0.80) // quantum * quantum)  # Reserve20% for thermal/runtime variation.
            manifest.update({"device": selected, "total_frames": frames, "per_run_seconds": per_run, "status": "training"})  # Freeze these before independent runs start.
            write_json(manifest_path, manifest)  # All six runs use this exact frozen frame target.
            LOGGER.info("suite frozen device=%s frames_per_run=%d seconds_per_run=%.1f power=%s", selected, frames, per_run, manifest["power_sensitivity"])  # Show assumptions alongside the allocation.
        for condition, seed in RUN_ORDER:  # Counterbalanced order is pre-specified above.
            name = f"{condition}_seed{seed}"  # Stable identity connects source, checkpoint, evaluation, and report.
            path = directory / name  # One isolated artifact directory per independent run.
            result_path = path / "result.json"  # A completed/timed-out run is never silently retrained.
            if result_path.exists():  # Resume reuses previously completed evidence.
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                available = remaining_seconds(manifest)  # Honor the original absolute deadline at every phase boundary.
                if available <= 30:  # Leave enough time for clean final bookkeeping.
                    manifest["status"] = "budget_exhausted"  # Partial experiments remain explicitly incomplete.
                    break  # Do not manufacture missing independent runs.
                checkpoint = path / "model.pt"  # Resume an interrupted run only if a valid checkpoint exists.
                result = train_run(TrainConfig(output_dir=str(path), condition=condition, seed=seed, num_envs=num_envs,
                                               total_frames=manifest["total_frames"], max_seconds=min(manifest["per_run_seconds"], available - 15),
                                               device=manifest["device"], resume=str(checkpoint) if checkpoint.exists() else None),
                                   deadline_monotonic=time.monotonic() + remaining_seconds(manifest) - 10)  # Local: transient suite deadline cannot be overridden by restored per-run settings.
            manifest["runs"][name] = result  # Retain timeouts as results rather than hiding unsuccessful runs.
            write_json(manifest_path, manifest)  # A later failure cannot erase completed training evidence.
        manifest["status"] = "evaluation" if len(manifest["runs"]) == 6 else "budget_exhausted"  # Separate implementation completion from experiment completion.
        write_json(manifest_path, manifest)  # External monitoring can inspect the current phase.
        for name in manifest["runs"]:  # Evaluate every completed trained model, not the best-looking one.
            path = directory / name  # Both initial and final actor checkpoints belong to this run.
            for model_name, evaluation_name in (("initial.pt", "initial_evaluation.json"), ("model.pt", "evaluation.json")):  # Publish before/after evidence.
                evaluation_path = path / evaluation_name  # Separate outputs prevent overwriting initial performance.
                if evaluation_path.exists() and json.loads(evaluation_path.read_text(encoding="utf-8")).get("status") == "complete":  # Skip only genuinely complete tournaments.
                    continue  # No repeated final-test sampling to obtain favorable scores.
                if remaining_seconds(manifest) <= 15:  # Stop cleanly at the original total deadline.
                    manifest["status"] = "budget_exhausted"  # Reports must show which evaluations are missing.
                    break  # Keep already saved raw rounds from earlier checkpoints.
                evaluate_policy(PolicyCheckpoint(path / model_name), evaluation_path, pairs, num_envs,
                                manifest["test_seed_base"], deadline=time.monotonic() + remaining_seconds(manifest) - 10,
                                resume=evaluation_path.exists())  # Local: continue validated partial evidence; test worlds never train or promote policies.
        trained = len(manifest["runs"]) == 6 and all(run.get("complete", False) for run in manifest["runs"].values())  # Local: time-limited unequal-frame runs cannot satisfy the frozen experiment budget.
        complete = trained and all((directory / name / "evaluation.json").exists() and json.loads((directory / name / "evaluation.json").read_text(encoding="utf-8")).get("status") == "complete" and (directory / name / "initial_evaluation.json").exists() and json.loads((directory / name / "initial_evaluation.json").read_text(encoding="utf-8")).get("status") == "complete" for name in manifest["runs"])  # Require every planned before/after tournament as well as complete training.
        manifest["status"] = "complete" if complete else "budget_exhausted"  # Report missing evidence honestly.
        manifest["finished_utc"] = datetime.now(UTC).isoformat()  # Timestamp the measured outcome, not a planned end.
        recommendation = directory / "selfplay_seed111" / "model.pt"  # Preselect deployment seed before seeing test results.
        if recommendation.exists():  # Even a weak policy remains available for user inspection.
            write_json(directory / "recommended.json", {"checkpoint": str(recommendation), "selection": "pre-specified selfplay seed111; not selected using final-test performance"})  # Avoid evaluation leakage through model selection.
        write_json(manifest_path, manifest)  # Save the final execution state before returning.
        LOGGER.info("suite finished status=%s runs=%d", manifest["status"], len(manifest["runs"]))  # Notify meaningful phase completion.
        return manifest  # Reporting/publishing remain explicit subsequent application phases.
    except Exception as error:  # Preserve the failing phase and traceback in durable diagnostics.
        manifest["status"] = "failed"  # Monitoring can distinguish failure from an unchanged running process.
        manifest["error"] = repr(error)  # Full traceback is also emitted by the root CLI logger.
        write_json(manifest_path, manifest)  # Do not lose experiment identity when a dependency fails.
        raise  # Keep a nonzero process exit code for unattended execution.
