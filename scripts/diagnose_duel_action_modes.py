"""Local: launch the registered action-mode diagnosis; global: a small wrapper delegates provenance, execution and analysis."""

import argparse  # Local: use an explicit published configuration.
from pathlib import Path  # Local: log only in the project's diagnostic directory.

from ministrike.experiments.action_mode_diagnostic import run_action_modes  # Local: reusable phase orchestration.
from ministrike.records import configure_logging  # Local: full timestamped terminal/file logs.


def main():
    """Local: parse, log, execute; global: imports never launch matches or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/duel-action-modes-v1.json")
    args = parser.parse_args()
    configure_logging(Path("runs/duel-fixed-v1/diagnostics/action-modes-v1/logs"))
    run_action_modes(args.config)


if __name__ == "__main__":
    main()  # Local: execution requires the explicit command.
