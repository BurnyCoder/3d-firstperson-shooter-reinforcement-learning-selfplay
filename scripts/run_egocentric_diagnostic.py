"""Local: verify real resume, train a registered representation intervention, evaluate; global: no automatic confirmation."""

import argparse  # Local: recovery is explicit, preserving source and budget identity.
from pathlib import Path  # Local: project-local logging.

from ministrike.experiments.registration import read_json, remaining_budget
from ministrike.experiments.resume_verification import verify_real_resume  # Local: use a tiny separate correctness control before long training.
from ministrike.experiments.staged import run_group  # Local: existing hidden-worker scheduling, telemetry and recovery.
from ministrike.experiments.egocentric_diagnostic import evaluate_egocentric  # Local: frozen controls and persisted evaluation cutoff.
from ministrike.records import configure_logging


def main():
    """Local: call clearly separated phases; global: every optimizer uses a pushed registration and the existing absolute clock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/duel-egocentric-R1.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    configure_logging(Path("runs/duel-fixed-v1/diagnostics/egocentric-v1/logs"))
    if not args.resume and remaining_budget(phase="A-DIAG") < config["runs"][0]["max_seconds"] + 1050:
        raise TimeoutError("Insufficient existing window for verification, diagnostic training and validation")
    verify_real_resume("configs/duel-resume-control.json")
    result = run_group(args.config, [0], "diagnostics/egocentric-R1", allow_resume=args.resume)
    if result["status"] != "complete" or not all(row.get("complete") for row in result.get("results", [])):
        raise RuntimeError("Preserved incomplete representation run; inspect before any recovery")
    evaluate_egocentric(args.config)


if __name__ == "__main__":
    main()  # Local: importing the entry point does not run an experiment.
