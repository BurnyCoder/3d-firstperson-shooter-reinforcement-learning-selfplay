"""Local: reproduce completed observer-coordinate reports; global: no gameplay, optimizer, or final-test access."""

import argparse  # Local: paths are explicit and reproducible.
from pathlib import Path  # Local: logs stay in this project.
from ministrike.experiments.egocentric_reporting import publish_egocentric  # Local: audit/copy/plot phases share existing analysis.
from ministrike.records import configure_logging  # Local: preserve full timestamped terminal/file output.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/duel-fixed-v1/diagnostics/egocentric-v1")
    parser.add_argument("--control", default="runs/duel-fixed-v1/diagnostics/duration-entropy-v1/control")
    parser.add_argument("--output", default="reports/duel-fixed-v1/egocentric-v1")
    args = parser.parse_args()
    configure_logging(Path("runs/reporting/egocentric-v1"))
    publish_egocentric(args.input, args.control, args.output)  # Local: imports do not perform evidence writes.
