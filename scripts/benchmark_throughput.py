"""Local: bounded hardware sweep; global: compare throughput without contaminating the live learner.

Source: https://psutil.readthedocs.io/en/latest/#psutil.Process.suspend .
Run with uv run --with psutil python scripts/benchmark_throughput.py --pause-pid PID.
The active learner's original wall-clock budget continues while briefly suspended.
"""

import argparse  # Explicit target PID prevents pausing arbitrary machine processes.
import json  # Save full probe configuration and process outcomes.
from pathlib import Path  # Restrict generated evidence to the selected local output folder.
import subprocess  # Fresh processes isolate thread pools, RNGs and CUDA contexts.
import sys  # Use the same locked project interpreter for each calibration.
import time  # Enforce a bounded pause even if a benchmark hangs.

import psutil  # Maintained process API supports reversible Windows suspend/resume.

from ministrike.records import configure_logging, write_json  # Reuse UTC diagnostics and atomic artifacts.
import logging  # Emit each phase start/finish to terminal and file.


def run_group(root, name, configurations, deadline):
    """Local: run independent calibration workers together. Global: measure complete process concurrency."""
    workers = []  # Keep handles for unconditional cleanup of only our own benchmark children.
    started = time.monotonic()  # Process wall time includes startup and warmup, unlike steady-state phase timing.
    try:
        for index, (arenas, threads, condition, profile) in enumerate(configurations):  # Each item is a separate learner.
            output = root / f"{name}_{index}"  # Preserve every candidate, including failures.
            output.mkdir(parents=True, exist_ok=False)  # Never overwrite earlier probe artifacts.
            stream = (output / "process.log").open("w", encoding="utf-8")  # Full child output remains untruncated.
            command = [sys.executable, "-u", "-m", "ministrike.experiments.performance",
                       "--output", str(output), "--num-envs", str(arenas), "--torch-threads", str(threads),
                       "--condition", condition, "--seed", str(8002 + index), "--updates", "4"]  # Same algorithm defaults across candidates.
            if profile:  # Profile overhead is excluded from the child's timed throughput interval.
                command.append("--profile-collection")  # Request one additional collection call graph.
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)  # No shell interpolation or visible helper window.
            workers.append((process, stream, output))  # Retain every handle before waiting for any worker.
        for process, _, output in workers:  # Concurrent children are already all running.
            code = process.wait(timeout=max(1, deadline - time.monotonic()))  # Global watchdog bounds the suspension interval.
            if code:  # Failed trials remain visible and do not become favorable timing results.
                raise RuntimeError(f"Calibration failed ({code}); inspect {output / 'process.log'}")
        return {"name": name, "process_wall_seconds": time.monotonic() - started,
                "workers": [json.loads((output / "performance.json").read_text(encoding="utf-8"))
                            for _, _, output in workers]}  # Retain raw per-phase results for correct aggregate analysis.
    finally:
        for process, stream, _ in workers:  # Clean up only benchmark processes created above.
            if process.poll() is None:  # An expired watchdog cannot leave GPU workloads running.
                process.kill()  # These are disposable calibration workers, never the live learner.
                process.wait()  # Release process resources before restoring the main experiment.
            stream.close()  # Preserve complete written diagnostic output.


def main():
    """Local: suspend, benchmark and resume reliably. Global: frozen learner state is never edited."""
    parser = argparse.ArgumentParser(description=__doc__)  # The target and duration are explicit operator controls.
    parser.add_argument("--pause-pid", type=int, required=True)  # Root verifies this is the active training process first.
    parser.add_argument("--output", type=Path, default=Path("runs/performance/sweep1"))  # Project-local generated evidence.
    parser.add_argument("--max-seconds", type=float, default=180)  # Count suspension against the original experiment deadline.
    sweep = parser.add_mutually_exclusive_group()  # Each invocation answers one hardware question with bounded extra compute.
    sweep.add_argument("--wide", action="store_true")  # Explore four/six simultaneous learners after the small initial sweep.
    sweep.add_argument("--thread-sweep", action="store_true")  # Compare one/two CPU threads with all six independent learners.
    args = parser.parse_args()  # Validate CLI shape before touching a running process.
    configure_logging(args.output / "logs")  # Mirror orchestration progress to timestamped file and terminal.
    process = psutil.Process(args.pause_pid)  # Resolve one known process rather than an unbounded process-name search.
    command = process.cmdline()  # Verify identity from the authorized command before suspension.
    if "ministrike.app" not in command or "--suite" not in command or Path(process.cwd()).resolve() != Path.cwd().resolve():  # Refuse other projects and training modes.
        raise ValueError("pause-pid must identify this project's running MiniStrike suite")
    configurations = [("e16_t4", [(16, 4, "selfplay", True)]),
                      ("e16_t1", [(16, 1, "selfplay", False)]),
                      ("e16_t2", [(16, 2, "selfplay", False)]),
                      ("e64_t1", [(64, 1, "selfplay", False)]),
                      ("parallel_t1", [(16, 1, "selfplay", False), (16, 1, "fixed", False)]),
                      ("parallel_t4", [(16, 4, "selfplay", False), (16, 4, "fixed", False)])]  # Compare measured work, not GPU utilization alone.
    if args.wide:  # Keep the follow-up bounded rather than repeating already measured candidates.
        configurations = [(f"parallel_{count}", [(16, 1, "selfplay" if index % 2 == 0 else "fixed", False)
                                                for index in range(count)]) for count in (4, 6)]  # Balanced conditions share the same GPU in each group.
    if args.thread_sweep:  # Hold concurrent model count and per-model learning settings constant.
        configurations = [(f"six_threads_{threads}", [(16, threads, "selfplay" if index % 2 == 0 else "fixed", False)
                                                     for index in range(6)]) for threads in (1, 2)]  # Observe CPU thread overhead under actual shared hardware pressure.
    started = time.monotonic()  # The pause duration is evidence and consumes the unchanged wall allowance.
    receipt = {"paused_pid": process.pid, "groups": [], "purpose": "hardware calibration only"}
    suspended = []  # Restore every successfully suspended process even if a later suspension fails.
    try:
        process.suspend()  # Prevent a suite parent from dispatching more work while its current descendants are enumerated.
        suspended.append(process)  # The finally block owns this reversible pause from this point onward.
        for child in process.children(recursive=True):  # Concurrent suites have independent collector/optimizer processes below the verified parent.
            child.suspend()  # Preserve each learner's live rounds, weights and RNGs without checkpoint rollback.
            suspended.append(child)  # A partial pause failure still releases all earlier successfully suspended children.
        receipt["paused_process_ids"] = [item.pid for item in suspended]  # Retain exact process scope for recovery/auditing.
        write_json(args.output / "sweep.json", receipt)  # Record the pause before starting disposable optimizer processes.
        logging.info("Paused verified suite processes %s for at most %.0f seconds", receipt["paused_process_ids"], args.max_seconds)
        for name, candidates in configurations:  # Execute stable candidate order, retaining every result.
            logging.info("Benchmark group %s", name)  # Each short phase remains visible to the user.
            receipt["groups"].append(run_group(args.output, name, candidates, started + args.max_seconds))
            write_json(args.output / "sweep.json", receipt)  # A later failure cannot erase completed measurements.
    finally:
        for item in reversed(suspended):  # Resume collectors before the parent resumes result bookkeeping.
            try:
                item.resume()  # Always release the live experiment, including on timeout or failed calibration.
            except psutil.NoSuchProcess:  # Preserve evidence if an external interruption removed a paused process.
                logging.error("Paused process %d disappeared before resume", item.pid)  # Do not conceal an independently interrupted learner.
        receipt["suspension_seconds"] = time.monotonic() - started  # Record actual time charged to the original run.
        write_json(args.output / "sweep.json", receipt)  # Preserve incomplete sweeps rather than silently retrying them.
        logging.info("Resumed suite process %d after %.3f seconds", process.pid, receipt["suspension_seconds"])


if __name__ == "__main__":  # Imports never suspend processes or start new optimization.
    main()  # Execute only on the operator's explicit benchmark invocation.
