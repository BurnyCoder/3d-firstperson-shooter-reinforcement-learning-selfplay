"""Local: bounded calibration/train/evaluate phases. Global: freeze a fair six-run comparison before learning.

Method sources: https://arxiv.org/abs/1709.06560 and https://arxiv.org/abs/1806.08295 .
Process scheduling: https://docs.python.org/3.12/library/concurrent.futures.html#processpoolexecutor .
The suite compares a small number of independent training seeds; it never trains until significance.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed  # Separate seed processes overlap CPU collection and GPU optimization.
from contextlib import nullcontext  # Reuse one result-persistence loop when serial execution needs no executor.
import json  # Resume the exact manifest instead of reselecting favorable settings.
import logging  # Keep complete phase progress in the application's timestamped log.
import multiprocessing  # Explicit spawn is portable to Windows and avoids inheriting initialized CUDA state.
import time  # Use monotonic time for in-process deadlines.
from datetime import UTC, datetime, timedelta  # Persist an absolute deadline across crashes/restarts.
from pathlib import Path  # All run directories remain under the requested suite folder.

import torch  # Discover supported optimization devices without assuming GPU speedups.

from ministrike.experiments.analysis import power_sensitivity  # Pre-training conditional sample-size calculation.
from ministrike.experiments.evaluation import evaluate_policy  # Use the exact public evaluation routine.
from ministrike.learning import PolicyCheckpoint, TrainConfig, calibrate, train_run  # Existing optimization implementation.
from ministrike.records import configure_logging, source_revision, write_json  # Per-worker logs and parent-only atomic manifests retain evidence.

LOGGER = logging.getLogger(__name__)  # Root CLI installs terminal and file handlers.
RUN_ORDER = [("selfplay", 111), ("fixed", 111), ("fixed", 222), ("selfplay", 222), ("selfplay", 333), ("fixed", 333)]  # Counterbalance sequential condition order.


def remaining_seconds(manifest: dict) -> float:
    """Local: calculate the original deadline's remaining allowance. Global: resume never grants more compute."""
    return max(0.0, (datetime.fromisoformat(manifest["deadline_utc"]) - datetime.now(UTC)).total_seconds())  # Include time since a crash or machine suspension.


def train_one(manifest: dict, condition: str, seed: int) -> dict | None:
    """Local: execute one immutable experiment identity. Global: the same picklable helper serves serial and concurrent schedules."""
    path = Path(manifest["output_dir"]) / f"{condition}_seed{seed}"  # Isolate checkpoints, metrics, and logs for each independent model.
    result_path = path / "result.json"  # A final result, including a timeout, is retained rather than retrained.
    if result_path.exists():  # Repeated suite invocations must not duplicate completed experimental work.
        return json.loads(result_path.read_text(encoding="utf-8"))  # Reuse the full recorded outcome unchanged.
    available = remaining_seconds(manifest)  # Always enforce the original overall suite deadline.
    if "training_deadline_utc" in manifest:  # A recorded scheduling amendment can reserve the remaining window for evaluation.
        available = min(available, max(0.0, (datetime.fromisoformat(manifest["training_deadline_utc"]) - datetime.now(UTC)).total_seconds()))
    if available <= 30:  # A queued task may begin after the training window has expired.
        return None  # Missing training remains missing evidence; never fabricate a result or extend the deadline.
    deadline = time.monotonic() + available - 10  # Fix the shared endpoint before logger or trainer initialization consumes time.
    if manifest.get("execution_workers", 1) > 1:  # Spawned workers do not inherit the parent's logging handlers.
        configure_logging(path / "logs")  # Each process records complete timestamped diagnostics and also streams to the terminal.
    checkpoint = path / "model.pt"  # An interrupted run continues only from its own saved optimizer and random state.
    config = TrainConfig(output_dir=str(path), condition=condition, seed=seed, num_envs=manifest["num_envs"],
                         total_frames=manifest["total_frames"], max_seconds=min(manifest["per_run_seconds"], available - 15),
                         device=manifest["device"], source_revision=manifest["source_revision"],
                         torch_threads=manifest.get("torch_threads", 4), resume=str(checkpoint) if checkpoint.exists() else None)
    try:  # Worker exceptions must retain a full traceback even before the parent receives the failed future.
        return train_run(config, deadline_monotonic=deadline)  # The common deadline also bounds restored checkpoint allowances.
    except Exception:  # Preserve diagnostics in this experiment's log without replacing its last valid checkpoint.
        LOGGER.exception("suite worker failed condition=%s seed=%d", condition, seed)  # Full exception output is not truncated.
        raise  # The parent records failed suite status and never invents completed training evidence.


def record_failure(manifest: dict, path: Path, error: Exception) -> None:
    """Local: preserve an orchestration error before cleanup waits. Global: monitoring never mistakes a failed suite for healthy progress."""
    manifest["status"] = "failed"  # Distinguish errors from unchanged running or exhausted-budget state.
    manifest["error"] = repr(error)  # Worker and root logs retain the accompanying full traceback.
    write_json(path, manifest)  # The parent remains the sole writer of the suite's durable result ledger.


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
        workers = manifest.get("execution_workers", 1)  # Old manifests preserve their original serial execution by default.
        if type(workers) is not int or not 1 <= workers <= 6:  # At most the six independent planned models may run together.
            raise ValueError("execution_workers must be an integer from 1 through 6")  # Reject malformed scheduling amendments before dispatch.
        threads = manifest.get("torch_threads", 4)  # Preserve the original CPU thread setting unless an amendment explicitly changes it.
        if type(threads) is not int or threads < 1:  # Thread counts are execution settings, not an invitation to change learning batches.
            raise ValueError("torch_threads must be a positive integer")  # Invalid process settings cannot silently fall back.
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
                                                      num_envs=num_envs, device=candidate, source_revision=manifest["source_revision"],
                                                      max_seconds=min(180, remaining_seconds(manifest))))  # Includes actual PPO updates/checkpoint writes under the frozen suite source.
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
        manifest["status"] = "training"  # Resuming an audited execution handoff now represents active training work.
        write_json(manifest_path, manifest)  # Publish the active phase before any serial or spawned optimizer task starts.
        execution = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else nullcontext()  # Parallelize independent models, never temporal PPO trajectories.
        with execution as executor:  # Join worker processes before held-out evaluation begins.
            futures = {}  # Serial execution has no queued tasks to cancel if a run fails.
            if executor is not None:  # Only an explicit recorded scheduling amendment enables parallel workers.
                frozen = {key: value for key, value in manifest.items() if key != "runs"}  # Worker inputs exclude the mutable parent-owned result ledger.
                futures = {executor.submit(train_one, frozen, condition, seed): f"{condition}_seed{seed}" for condition, seed in RUN_ORDER}  # Submit each planned identity once in its recorded order.
                results = ((futures[future], future.result()) for future in as_completed(futures))  # Persist ready work without waiting for an earlier slow run.
            else:
                results = ((f"{condition}_seed{seed}", train_one(manifest, condition, seed)) for condition, seed in RUN_ORDER)  # Default serial mode uses the same budget and resume rules.
            try:  # A failed future must become visible before executor cleanup waits for running siblings.
                for name, result in results:  # Only this parent process writes suite.json, preventing concurrent file replacement races.
                    if result is not None:  # Expired queued tasks produce no synthetic result and remain visibly incomplete.
                        manifest["runs"][name] = result  # Retain every completed or time-limited outcome under its exact identity.
                        write_json(manifest_path, manifest)  # Completion order cannot erase another run's evidence.
            except Exception as error:  # Keep failed status durable while already-running workers finish within their common deadline.
                record_failure(manifest, manifest_path, error)  # Monitoring receives failure immediately, before the context may block.
                for future in futures:  # Prevent queued independent runs from starting after an execution failure.
                    future.cancel()  # Already-running siblings retain their own valid checkpoints/results for later resume.
                raise  # The outer handler preserves the same error and the CLI exits unsuccessfully.
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
        record_failure(manifest, manifest_path, error)  # Covers calibration/evaluation errors as well as failed worker futures.
        raise  # Keep a nonzero process exit code for unattended execution.
