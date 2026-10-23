"""Local: one evaluation entry point; global: preserve the registered validation-before-test procedure."""

from pathlib import Path  # Local: full logs stay with the experiment.
from ministrike.experiments.confirmation import evaluate_confirmation
from ministrike.records import configure_logging


if __name__ == "__main__":
    configure_logging(Path("runs/duel-fixed-v1/confirmation/evaluation-logs"))
    evaluate_confirmation()  # Local: imports perform no gameplay, training or publication.
