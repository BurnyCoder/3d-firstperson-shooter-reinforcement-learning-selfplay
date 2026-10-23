"""Local: regenerate confirmation evidence; global: reporting never launches another training or evaluation run."""

import argparse  # Local: allow raw run directories or previously copied evidence.
from pathlib import Path  # Local: explicit report and log destinations.
from ministrike.experiments.confirmation_reporting import publish_confirmation_evidence
from ministrike.records import configure_logging

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", default="runs/duel-fixed-v1/confirmation", type=Path)
parser.add_argument("--output", default="reports/duel-fixed-v1/confirmation", type=Path)
arguments = parser.parse_args()
configure_logging(Path("runs/report-logs"))  # Local: capture complete audit failures and exported artifact identities.
publish_confirmation_evidence(arguments.input, arguments.output)
