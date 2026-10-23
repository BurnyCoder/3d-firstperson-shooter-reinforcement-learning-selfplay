"""Local: publish completed diagnostics; global: no optimizer, new games or final-test access."""

import argparse  # Local: explicit paths support reproducible local evidence export.
from pathlib import Path  # Local: logs remain project-local.

from ministrike.experiments.duration_reporting import publish_duration_evidence  # Local: reusable audit/plot phases.
from ministrike.records import configure_logging  # Local: timestamp all reporting output.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/duel-fixed-v1/diagnostics/duration-entropy-v1")
    parser.add_argument("--output", default="reports/duel-fixed-v1/duration-entropy-v1")
    args = parser.parse_args()
    configure_logging(Path("runs/reporting/duration-entropy-v1"))
    publish_duration_evidence(args.input, args.output)  # Local: no work occurs when this wrapper is imported.
