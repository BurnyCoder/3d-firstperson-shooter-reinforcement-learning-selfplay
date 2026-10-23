"""Local: run two registered independent diagnostics serially; global: reuse calibrated scheduling and source/budget gates.

Protocol: docs/amendments/2026-10-23-duration-entropy-diagnostic.md.
Process supervision, telemetry and recovery remain in ministrike.experiments.staged.
"""

import argparse  # Local: recovery is explicit and retains the original identities and deadlines.
from pathlib import Path  # Local: logs/configuration remain in this project.

from ministrike.experiments.registration import read_json, remaining_budget  # Local: one Stage A clock, never a new allowance.
from ministrike.experiments.staged import run_group  # Local: reuse immutable config checks, hidden workers and failure preservation.
from ministrike.experiments.duration_entropy import evaluate_duration_entropy  # Local: evaluate only after both trainers have stopped.
from ministrike.records import configure_logging  # Local: mirror complete supervisor diagnostics to terminal/file.


def main():
    """Local: parse, verify time, supervise; global: no automatic confirmation or reward selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/duel-duration-entropy-v1.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    configure_logging(Path("runs/duel-fixed-v1/diagnostics/duration-entropy-v1/logs"))
    if not args.resume and remaining_budget(phase="A-DIAG") < sum(row["max_seconds"] for row in config["runs"]) + 930:
        raise TimeoutError("Insufficient existing diagnostic window for both registered runs and validation reserve")
    for index, name in enumerate(config["groups"]):
        result = run_group(args.config, [index], name, allow_resume=args.resume)
        if result["status"] != "complete" or not all(row.get("complete") for row in result.get("results", [])):
            raise RuntimeError(f"Preserved incomplete diagnostic group: {name}; inspect before any recovery")
    evaluate_duration_entropy(args.config)  # Local: immutable endpoints and one 900-second validation cutoff; no final-test access.


if __name__ == "__main__":
    main()  # Local: imports never start an experiment.
