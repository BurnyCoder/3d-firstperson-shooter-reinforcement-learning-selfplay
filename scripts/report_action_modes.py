"""Local: audit and export the frozen diagnostic; global: reporting never launches gameplay or optimization."""

from pathlib import Path  # Local: the shared logger receives a project-local directory.

from ministrike.experiments.mode_reporting import publish_mode_evidence  # Local: readable orchestration over reusable phases.
from ministrike.records import configure_logging  # Local: retain full timestamped reporting errors.

if __name__ == "__main__":
    configure_logging(Path("runs/duel-fixed-v1/diagnostics/action-modes-v1/report-logs"))
    publish_mode_evidence("runs/duel-fixed-v1/diagnostics/action-modes-v1", "reports/duel-fixed-v1")
