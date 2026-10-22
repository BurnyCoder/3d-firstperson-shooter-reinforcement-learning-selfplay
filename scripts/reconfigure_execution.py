"""Local: audited checkpoint handoff; global: change hardware scheduling without restarting learning.

Sources: https://psutil.readthedocs.io/ and https://docs.python.org/3/library/datetime.html .
Run with uv run --with psutil python scripts/reconfigure_execution.py --pid PID --workers 6.
This operator tool preserves the original absolute deadline and all model/optimizer/RNG states.
"""

import argparse  # Explicit operator settings keep process selection and concurrency reviewable.
from datetime import UTC, datetime, timedelta  # Compute the original reserved evaluation boundary.
import hashlib  # Fingerprint the unchanged archived checkpoint for the execution audit.
import json  # Read exact frozen settings and completed update records.
import logging  # Record every handoff phase in timestamped terminal/file logs.
from pathlib import Path  # Bound every mutation to this project and the selected suite.
import shutil  # Preserve original evidence before replacing runtime metadata.
import time  # Bound waiting for the next ordinary periodic checkpoint.

import psutil  # Suspend the verified producer before atomically handing off its checkpoint.

from ministrike.learning.checkpoint import load_checkpoint, save_checkpoint  # Reuse production weights-only state serialization.
from ministrike.records import configure_logging, source_revision, timestamp, write_json  # Shared provenance and atomic JSON.


def await_checkpoint(process, run, timeout=90):
    """Local: stop just after a persisted update. Global: resume from an exact existing optimizer boundary."""
    started = datetime.now(UTC)  # Require fresh progress rather than accepting a stale pre-pause log row.
    deadline = time.monotonic() + timeout  # Never wait indefinitely for a broken producer.
    while time.monotonic() < deadline:  # The learner continues normal training until a checkpoint boundary.
        rows = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()  # Existing logs are small and retained fully.
        try:
            row = json.loads(rows[-1])  # A partial last append is retried rather than treated as a checkpoint.
        except (json.JSONDecodeError, IndexError):
            time.sleep(0.05)  # Yield briefly to the running producer before retrying.
            continue
        recorded = datetime.strptime(row["timestamp_utc"], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC)
        if row["updates"] % 10 == 0 and recorded > started:  # Production snapshots occur every ten completed updates.
            process.suspend()  # Freeze all producer threads before reading checkpoint bytes and the final log row.
            if (run / "result.json").exists():  # A completed run is immutable evidence, not a target for this handoff.
                raise RuntimeError("Run completed before handoff; original suite will resume unchanged")
            try:
                frozen_row = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])  # Re-read after suspension to detect later completed updates.
            except (json.JSONDecodeError, IndexError):
                process.resume()  # An interrupted partial append needs its producer running before a retry.
                continue
            state = load_checkpoint(run / "model.pt")  # Atomic production saves expose only complete files.
            if frozen_row == row and state["frames"] == row["frames"]:  # Checkpoint and final metric must identify one exact persisted boundary.
                return state, row  # Caller retains the producer suspension until the audited handoff completes.
            process.resume()  # A raced checkpoint observation is harmless; wait for the next matching boundary.
        time.sleep(0.05)  # Low-overhead polling leaves CPU time to the authorized learner.
    raise TimeoutError("No fresh matching periodic checkpoint within the handoff window")


def main():
    """Local: validate, archive, amend and stop. Global: no learning state or step budget is reset."""
    parser = argparse.ArgumentParser(description=__doc__)  # This is an explicit operator action, never normal training startup.
    parser.add_argument("--pid", type=int, required=True)  # Root verifies the active producer before invocation.
    parser.add_argument("--workers", type=int, choices=range(1, 7), required=True)  # At most the six registered independent learners.
    parser.add_argument("--threads", type=int, default=1)  # Benchmarked intra-op threads, not a neural hyperparameter.
    parser.add_argument("--suite", type=Path, default=Path("runs/suite"))  # The existing suite identity remains unchanged.
    parser.add_argument("--run", default="selfplay_seed111")  # The single currently active sequential run to hand off.
    args = parser.parse_args()  # Reject malformed settings before touching the existing process.
    root, directory = Path.cwd().resolve(), args.suite.resolve()  # Current working directory defines the authorized workspace.
    run = (directory / args.run).resolve()  # Resolve traversal before any backup or replacement.
    if not directory.is_relative_to(root) or not run.is_relative_to(directory) or args.threads < 1:
        raise ValueError("Suite/run must stay inside this project and threads must be positive")
    configure_logging(directory / "handoff-logs")  # Preserve operator diagnostics beside experiment artifacts.
    process = psutil.Process(args.pid)  # Target only the caller's verified process ID.
    command = process.cmdline()  # Validate identity without inspecting unrelated applications.
    if "ministrike.app" not in command or "--suite" not in command or Path(process.cwd()).resolve() != root:
        raise ValueError("PID is not this project's running suite")
    manifest = json.loads((directory / "suite.json").read_text(encoding="utf-8"))  # Freeze original deadlines before handoff.
    if manifest.get("execution_workers", 1) != 1 or process.children():  # Suspending a parent never suspends independent learner children.
        raise ValueError("This handoff supports only a single sequential producer with no child processes")
    if (run / "result.json").exists():  # Never amend a finished outcome or interrupt a subsequent run using an older path.
        raise ValueError("The selected run already finished; this handoff is no longer applicable")
    training_deadline = datetime.fromisoformat(manifest["started_utc"]) + timedelta(hours=manifest["max_hours"] * 5 / 6)
    original_cap = manifest["per_run_seconds"]  # Retain the previous sequential allocation in the audit record.
    amended_cap = min(original_cap * args.workers, (training_deadline - datetime.now(UTC)).total_seconds())  # Parallel jobs overlap in wall time, within the original training window.
    if amended_cap <= 0:  # Never borrow from the originally reserved evaluation time.
        raise RuntimeError("The original training allocation is already exhausted")
    stopped = False  # Failure cleanup must release a still-running suspended producer.
    try:
        state, row = await_checkpoint(process, run)  # Preserve an exact model, optimizer, simulator and RNG boundary.
        archive = directory / "execution_amendments" / timestamp()  # Keep the original bytes and full metrics permanently.
        archive.mkdir(parents=True)  # This directory is new and never overwrites earlier evidence.
        for source in (directory / "suite.json", run / "model.pt", run / "config.json", run / "metrics.jsonl"):
            shutil.copy2(source, archive / source.name)  # Back up complete originals before mutation.
        original_hash = hashlib.sha256((archive / "model.pt").read_bytes()).hexdigest()  # Tie the amendment to exact learned state.
        saved = datetime.strptime(state["saved_at_utc"], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC)
        state["elapsed_seconds"] += max(0.0, (datetime.now(UTC) - saved).total_seconds())  # Charge checkpoint/handoff time; never reset consumed budget.
        state["config"]["torch_threads"] = args.threads  # Only runtime scheduling changes, with all algorithm settings preserved.
        state["config"]["max_seconds"] = amended_cap  # The shared training deadline remains the ultimate limit.
        amendment = {"timestamp_utc": timestamp(), "reason": "User requested maximum practical GPU use; throughput-only concurrent calibration",
                     "execution_revision": source_revision(), "old_pid": args.pid, "run": args.run,
                     "frames": row["frames"], "updates": row["updates"], "original_checkpoint_sha256": original_hash,
                     "archive": str(archive), "workers_before": manifest.get("execution_workers", 1),
                     "workers_after": args.workers, "torch_threads": args.threads,
                     "per_run_seconds_before": original_cap, "per_run_seconds_after": amended_cap,
                     "training_deadline_utc": training_deadline.isoformat(), "overall_deadline_utc": manifest["deadline_utc"],
                     "unchanged": "weights, optimizer, simulator, RNGs, actor/critic architecture, PPO settings, arena count, rollout length, frozen total frames and hypothesis"}
        manifest.setdefault("execution_amendments", []).append(amendment)  # Preserve earlier protocol decisions alongside this explicit amendment.
        manifest.update(execution_workers=args.workers, torch_threads=args.threads, per_run_seconds=amended_cap,
                        training_deadline_utc=training_deadline.isoformat(), status="execution_reconfigured")
        process.kill()  # The complete archived checkpoint is now the sole authoritative continuation state.
        process.wait(timeout=10)  # Ensure the original producer cannot overwrite the amended checkpoint or manifest.
        stopped = True  # Cleanup must not attempt to resume a terminated producer.
        save_checkpoint(run / "model.pt", state)  # Atomic replacement modifies only the disclosed runtime metadata and elapsed accounting.
        write_json(run / "config.json", state["config"])  # On-disk config agrees with what train_run restores.
        write_json(directory / "suite.json", manifest)  # All six workers inherit the same scheduling settings and frozen frame budget.
        write_json(archive / "amendment.json", amendment)  # The original evidence and rationale remain together.
        logging.info("Execution handoff ready to resume: %s", json.dumps(amendment))  # Report the exact preserved boundary.
    finally:
        if not stopped and process.is_running():  # An error before termination must not strand the user's training process.
            process.resume()  # Return the original learner to normal execution when handoff failed early.


if __name__ == "__main__":  # Importing this operator tool never touches a running experiment.
    main()  # Only an explicit command performs the reviewed handoff.
