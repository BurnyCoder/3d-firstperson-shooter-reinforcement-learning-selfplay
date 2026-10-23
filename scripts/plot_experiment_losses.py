"""Local: plot existing experiment logs; global: reporting never starts or resumes optimizer work.

Matplotlib source references and data integrity checks live in the reusable loss_plots module.
"""

import argparse  # Local: explicit source/output paths prevent accidental discovery of unrelated runs.
from pathlib import Path  # Local: all outputs remain in the selected project directory.

from ministrike.experiments.loss_plots import save_loss_gallery  # Local: share the report pipeline's plotting phase.
from ministrike.experiments.evaluation_plots import plot_pilot_evaluations  # Local: show measured validation beside optimizer diagnostics.
from ministrike.records import configure_logging  # Local: retain timestamped terminal/file output.


def main():
    """Local: parse, log, plot; global: a small wrapper calls the reusable artifact-producing phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="**/metrics.jsonl")  # Local: explicitly select canonical runs when needed.
    args = parser.parse_args()
    configure_logging(Path("runs/logs"))
    save_loss_gallery(args.input, args.output, args.pattern)  # Local: no training/checkpoint mutation occurs.
    plot_pilot_evaluations(args.input, args.output)  # Local: produces figures only when all pilot endpoint records are complete.


if __name__ == "__main__":
    main()  # Local: importing this wrapper never writes files.
