"""Local: command-line dispatch. Global: one readable wrapper delegates each reusable phase.

Source: https://docs.python.org/3/library/argparse.html . No game, learning, or reporting logic lives here.
"""

import argparse  # Standard command parsing avoids an additional CLI framework.
import json  # Read the explicit post-training checkpoint recommendation.
import logging  # Report complete exceptions through the shared timestamped handlers.
from pathlib import Path  # Portable project-relative CLI destinations.

from ministrike.config import load_settings  # Load .env once before constructing parser defaults.
from ministrike.records import configure_logging  # Mirror all phase logs to file and terminal.


def create_parser(settings) -> argparse.ArgumentParser:
    """Local: declare the public CLI. Global: every mode remains directly scriptable and reproducible."""
    parser = argparse.ArgumentParser(description="MiniStrike: play, watch, and train 3D tactical bots")  # No command opens the two-choice launcher.
    commands = parser.add_subparsers(dest="command")  # Keep launch optional for human users.
    for mode in ("play", "watch"):  # Both visible modes accept identical checkpoint/replay settings.
        command = commands.add_parser(mode, help=f"{mode} a local 5v5 game")  # Explicit modes bypass the launcher.
        command.add_argument("--checkpoint", type=Path)  # Omission uses the suite recommendation when available.
        command.add_argument("--seed", type=int, default=settings.seed)  # Reproduce initial positions and simulation randomness.
        command.add_argument("--team", type=int, choices=(0, 1), default=0)  # Human slot belongs to the selected team.
        command.add_argument("--seconds", type=float)  # Bounded rendering supports automated visual verification.
        command.add_argument("--screenshot", type=Path)  # Finite viewers also save action replay evidence.
        command.add_argument("--first-person", action="store_true")  # Watch through a bot's eyes before optional live takeover.
    train = commands.add_parser("train", help="train one run or the pre-registered six-run suite")  # Actual reinforcement learning.
    train.add_argument("--suite", action="store_true")  # Opt in to independent self-play/control runs.
    train.add_argument("--output", type=Path)  # Defaults differ between individual and suite runs.
    train.add_argument("--condition", choices=("selfplay", "fixed"), default="selfplay")  # Single-run control condition.
    train.add_argument("--seed", type=int, default=settings.seed)  # Independent training initialization.
    train.add_argument("--num-envs", type=int, default=settings.num_envs)  # Batch arenas for efficient collection.
    train.add_argument("--frames", type=int, default=65536)  # Arena transitions, not fivefold player transitions.
    train.add_argument("--max-hours", type=float, default=settings.max_hours)  # Shared suite or individual upper bound.
    train.add_argument("--device", choices=("cpu", "cuda", "auto"), default=settings.device)  # Auto benchmarks both supported devices in a suite.
    train.add_argument("--resume", type=Path)  # Individual checkpoint or existing suite manifest.
    train.add_argument("--pairs", type=int, default=128)  # Freeze final-test sample count before suite execution.
    evaluate = commands.add_parser("evaluate", help="evaluate a frozen checkpoint on held-out seed pairs")  # Never updates weights.
    evaluate.add_argument("--checkpoint", type=Path, required=True)  # Evaluation target is always explicit.
    evaluate.add_argument("--output", type=Path, default=settings.output_dir / "evaluation.json")  # Full raw results and summary.
    evaluate.add_argument("--pairs", type=int, default=128)  # Equal number of paired worlds across opponents.
    evaluate.add_argument("--num-envs", type=int, default=settings.num_envs)  # Record batching because it determines seeded world streams.
    evaluate.add_argument("--seed-base", type=int, default=2_000_000)  # Final-test namespace differs from training and validation.
    report = commands.add_parser("report", help="render measured experiment reports and PDF")  # No invented placeholder performance.
    report.add_argument("--input", type=Path, default=settings.output_dir / "suite")  # Read saved artifacts only.
    report.add_argument("--output", type=Path, default=Path("reports"))  # Versionable evidence separate from bulky checkpoints.
    return parser  # Tests can inspect argument behavior without starting graphics or training.


def recommended_checkpoint(output_dir: Path) -> Path | None:
    """Local: read the explicit suite recommendation. Global: never quietly substitute an unrelated model."""
    path = output_dir / "suite" / "recommended.json"  # Suite publication writes this stable pointer.
    if not path.exists():  # Before training, the launcher labels its scripted demonstration honestly.
        return None  # The renderer handles this as an explicit baseline mode.
    checkpoint = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])  # Pointer is produced by the local experiment runner.
    if not checkpoint.is_file():  # Stale pointers should not silently fall back to another model.
        raise FileNotFoundError(f"Recommended checkpoint missing: {checkpoint}")  # User can supply --checkpoint explicitly.
    return checkpoint  # Both play and watch load this same trained actor.


def execute_phase(arguments, settings):
    """Local: dispatch one mode. Global: implementations stay behind small imported functions."""
    if arguments.command in (None, "play", "watch"):  # Renderers are imported only for visible interaction.
        from ministrike.view import run_view  # Headless commands never construct Ursina objects.
        checkpoint = getattr(arguments, "checkpoint", None) or recommended_checkpoint(settings.output_dir)  # Prefer an explicit user-selected model.
        if arguments.command is None:  # Human entry point offers the two requested post-training choices.
            from ministrike.view import run_launcher  # Standard desktop launcher opens before the game engine.
            return run_launcher(checkpoint=checkpoint, seed=settings.seed)  # Choose watch or human first-person play.
        return run_view(mode=arguments.command, checkpoint=checkpoint, seed=arguments.seed, team=arguments.team,
                        seconds=arguments.seconds, screenshot_path=arguments.screenshot,
                        spectator_first_person=arguments.first_person)  # One renderer serves watch, first-person spectating, and takeover.
    if arguments.command == "train":  # Delay expensive torch imports until a learning command needs them.
        if arguments.suite:  # The suite owns its total deadline and frozen protocol.
            from ministrike.experiments.suite import run_suite  # Keep experiment scheduling separate from optimization.
            return run_suite(arguments.output or settings.output_dir / "suite", arguments.max_hours, arguments.device,
                             arguments.num_envs, arguments.pairs, arguments.resume)  # Persist every chosen setting.
        from ministrike.learning import TrainConfig, train_run  # Reuse the same trainer as the suite.
        return train_run(TrainConfig(output_dir=str(arguments.output or settings.output_dir / "single"), condition=arguments.condition,
                                    seed=arguments.seed, num_envs=arguments.num_envs, total_frames=arguments.frames,
                                    max_seconds=arguments.max_hours * 3600, device="cpu" if arguments.device == "auto" else arguments.device,
                                    resume=str(arguments.resume) if arguments.resume else None))  # Single-run auto defaults conservatively to CPU.
    if arguments.command == "evaluate":  # Frozen evaluation is independently repeatable.
        from ministrike.experiments.evaluation import evaluate_policy  # Tournament logic is shared with the suite.
        from ministrike.learning import PolicyCheckpoint  # Load the exact actor under evaluation.
        return evaluate_policy(PolicyCheckpoint(arguments.checkpoint), arguments.output, arguments.pairs,
                               arguments.num_envs, arguments.seed_base)  # Test seed namespace is saved in results.
    from ministrike.experiments.reporting import create_reports  # The only remaining parser mode is report.
    return create_reports(arguments.input, arguments.output)  # Reports derive claims from saved measured artifacts.


def main() -> None:
    """Local: configure, parse, log, execute. Global: one import-safe application entry point."""
    settings = load_settings()  # Read project-local settings without searching parent directories.
    arguments = create_parser(settings).parse_args()  # Validate options before constructing simulation state.
    configure_logging(settings.output_dir / "logs", settings.log_level)  # Complete diagnostics begin before executing any phase.
    logging.info("command %s", vars(arguments))  # Arguments contain experiment controls, never tokens or credentials.
    try:  # Persist failure tracebacks while preserving the command's nonzero exit status.
        execute_phase(arguments, settings)  # Delegate all implementation details to the selected subsystem.
    except Exception:  # Expected failures still need complete durable diagnostics.
        logging.exception("MiniStrike command failed")  # Standard logging includes the full exception traceback.
        raise  # Automation can reliably detect unsuccessful execution.


if __name__ == "__main__":  # Importing the package must never launch a window or a training process.
    main()  # Windows-safe entry point also supports python -m ministrike.app.
