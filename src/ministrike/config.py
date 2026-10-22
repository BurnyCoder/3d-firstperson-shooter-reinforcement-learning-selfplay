"""Local context: parse .env defaults. Global context: one configuration source for CLI phases.

Source: https://bbc2.github.io/python-dotenv/ ; explicit CLI arguments override these defaults.
"""

import os  # Read process configuration without logging unrelated environment variables.
from dataclasses import dataclass  # Keep the small immutable runtime configuration explicit.
from pathlib import Path  # Resolve all output paths within the current project by default.

from dotenv import load_dotenv  # Reuse the standard .env parser rather than implementing one.


@dataclass(frozen=True)
class Settings:
    """Local: typed CLI defaults. Global: separate runtime settings from game balance."""

    output_dir: Path = Path("runs")  # Generated artifacts are ignored by Git until selected for publication.
    num_envs: int = 16  # Batch arenas together to amortize neural-network inference.
    device: str = "cpu"  # CPU is portable; benchmark CUDA explicitly before selecting it.
    max_hours: float = 12.0  # The entire experiment suite shares this wall-clock allowance.
    seed: int = 111  # Training seeds are disjoint from evaluation seed ranges.
    log_level: str = "INFO"  # Normal runs show every iteration and complete errors.


def load_settings() -> Settings:
    """Local: load only this project's .env. Global: never search other project folders."""
    load_dotenv(Path.cwd() / ".env", override=False)  # Existing process settings take precedence.
    return Settings(  # Convert text once, making malformed settings fail before launching work.
        output_dir=Path(os.getenv("CSRL_OUTPUT_DIR", "runs")),
        num_envs=int(os.getenv("CSRL_NUM_ENVS", "16")),
        device=os.getenv("CSRL_DEVICE", "cpu"),
        max_hours=float(os.getenv("CSRL_MAX_HOURS", "12")),
        seed=int(os.getenv("CSRL_SEED", "111")),
        log_level=os.getenv("CSRL_LOG_LEVEL", "INFO"),
    )
