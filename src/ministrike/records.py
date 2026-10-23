"""Local context: durable JSON and timestamped logs. Global context: auditable experiment outputs.

Sources: https://docs.python.org/3/library/logging.html and https://docs.python.org/3/library/os.html#os.replace
"""

import json  # Use a portable format for manifests and measured results.
import logging  # Reuse the standard logger across all package subsystems.
import os  # Atomically replace completed manifests on the same filesystem.
import subprocess  # Read the exact source revision through Git's public CLI.
import sys  # Stream user-visible messages immediately to the existing terminal.
import time  # Standard formatter timestamps are explicitly UTC.
from datetime import UTC, datetime  # ISO timestamps avoid host-timezone ambiguity.
from pathlib import Path  # Handle Windows paths without shell interpolation.


def timestamp() -> str:
    """Local: filesystem-safe UTC identifier. Global: disambiguate independent invocations."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")  # Microseconds prevent adjacent CLI runs colliding.


def replace_artifact(temporary: Path, destination: Path) -> None:
    """Local: tolerate brief sharing denial; global: never delete the last durable artifact.

    Windows replace can fail while another handle is open: https://bugs.python.org/issue46003 .
    Five retries over1.55s are an engineering bound; persistent access failures still raise.
    """
    for attempt in range(6):  # Local: one initial attempt and at most five retries, never an infinite wait.
        try:
            os.replace(temporary, destination)  # Local: retain the standard atomic replacement operation.
            return  # Local: success publishes the fully written file exactly once.
        except PermissionError:
            if attempt == 5:
                raise  # Local: preserve both old destination and completed temporary file for diagnosis.
            logging.warning("Artifact replacement denied; retry %s/5: %s", attempt + 1, destination)
            time.sleep(.05 * 2 ** attempt)  # Local: bounded backoff lets transient reader handles close.


def write_json(path: str | Path, value: object) -> None:
    """Local: replace a complete JSON file. Global: interrupted runs retain the last valid manifest."""
    destination = Path(path)  # Normalize the interface once for every caller.
    destination.parent.mkdir(parents=True, exist_ok=True)  # Parent creation is idempotent on resume.
    temporary = destination.with_suffix(destination.suffix + ".tmp")  # Keep the rename on one filesystem.
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")  # Reject invalid metrics.
    replace_artifact(temporary, destination)  # Readers see either the previous or next complete file.


def configure_logging(directory: Path, level: str = "INFO") -> Path:
    """Local: mirror complete log records to file and terminal. Global: no truncation of errors or metrics."""
    directory.mkdir(parents=True, exist_ok=True)  # Keep each invocation's transcript alongside its artifacts.
    path = directory / f"{timestamp()}.log"  # Distinguish restarts without overwriting earlier evidence.
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s | %(message)s")  # Include emitting subsystem.
    formatter.converter = time.gmtime  # UTC timestamps remain comparable across machines.
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")]  # Both handlers flush per record.
    for handler in handlers:  # Apply one format consistently to both destinations.
        handler.setFormatter(formatter)  # Tracebacks are retained by standard exception formatting.
    logging.basicConfig(level=level, handlers=handlers, force=True)  # Replace stale handlers when called programmatically.
    logging.captureWarnings(True)  # Dependency warnings enter the same persistent record.
    return path  # CLI summaries can identify the precise log file.


def source_revision() -> str:
    """Local: ask Git for HEAD. Global: tie results to the actual implementation used."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)  # Never invoke a shell.
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"  # Explicitly label missing provenance.
