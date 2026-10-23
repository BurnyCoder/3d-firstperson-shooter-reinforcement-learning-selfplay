"""Local: audit and export the frozen diagnostic; global: reporting never launches gameplay or optimization."""

from pathlib import Path  # Local: the shared logger receives a project-local directory.
import argparse  # Local: published raw inputs can be replayed in a fresh clone without checkpoints.

from ministrike.experiments.mode_reporting import publish_mode_evidence  # Local: readable orchestration over reusable phases.
from ministrike.records import configure_logging  # Local: retain full timestamped reporting errors.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/duel-fixed-v1/diagnostics/action-modes-v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/duel-fixed-v1"))
    args = parser.parse_args()
    configure_logging(Path("runs/duel-fixed-v1/diagnostics/action-modes-v1/report-logs"))
    publish_mode_evidence(args.input, args.output)
