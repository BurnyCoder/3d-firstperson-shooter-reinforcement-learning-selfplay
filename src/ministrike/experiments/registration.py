"""Local: publication and deadline gates; global: optimizer work cannot outrun registration.

Engineering contract: docs/duel-protocol.md. Standard-library references:
https://docs.python.org/3.12/library/subprocess.html and /datetime.html .
"""

from dataclasses import asdict  # Local: compare every execution setting with its published record.
from datetime import datetime, timedelta, timezone  # Local: wall-clock deadlines survive process restarts.
import hashlib  # Local: identify exact published configuration bytes.
import json  # Local: human-readable durable receipts.
from pathlib import Path  # Local: constrain the one authorized budget to this project.
import subprocess  # Local: verify source publication against the actual Git remote.

from ministrike.learning.config import TrainConfig  # Local: one default-resolution implementation.

ROOT = Path(__file__).resolve().parents[3]  # Local: package source is inside this checkout.
MANIFEST = ROOT / "runs/duel-fixed-v1/manifest.json"  # Local: no alternate directory can reset the authorized clock.


def git(*arguments):
    """Local: run Git without a shell; global: arguments/config paths cannot become shell code."""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()  # Local: errors fail closed.


def read_json(path):
    """Local: read explicit UTF-8 JSON; global: registration and runtime share the same parser."""
    return json.loads(Path(path).read_text(encoding="utf-8"))  # Local: do not search unrelated directories.


def training_preflight(config):
    """Local: verify status, parameters, publication and budget; global: never optimize unregistered settings."""
    if not config.run_config or not config.registration:
        raise ValueError("Stage A requires a published run_config and roadmap registration")
    path = (ROOT / config.run_config).resolve()  # Local: recorded config must be in this checkout.
    relative = path.relative_to(ROOT).as_posix()  # Local: reject external configuration paths.
    registry = read_json(ROOT / "configs/experiments.json")  # Local: stable runnable identities are versioned.
    entry = registry["experiments"].get(config.registration, {})
    if entry.get("status") != "runnable":
        raise ValueError("Registration is not runnable; publish its resolved prerequisites first")
    if git("status", "--porcelain", "--untracked-files=all", "src", "configs", "docs", "tests", "README.md", "AGENTS.md", "pyproject.toml", "uv.lock"):
        raise ValueError("Implementation, tests or protocol are not committed")  # Local: user scratch files elsewhere are untouched.
    source = git("rev-parse", "HEAD")  # Local: current execution identity must exist on the public remote.
    branch = git("branch", "--show-current")
    remote = git("ls-remote", "--heads", "origin", branch)  # Local: do not trust an obsolete remote-tracking ref.
    if not remote or remote.split()[0] != source:
        raise ValueError("Push this source revision before any experiment or optimizer benchmark")
    raw = path.read_bytes()  # Local: configuration hash accompanies the source identity in receipts.
    published = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if raw.replace(b"\r\n", b"\n") != published.replace(b"\r\n", b"\n"):
        raise ValueError("Configuration differs from the published source")
    frozen = read_json(path)  # Local: one file may register independent workers in a hardware cell or suite.
    actual = asdict(config)  # Local: mutable runtime fields are excluded; all learning/hardware settings stay frozen.
    ignored = {"resume", "source_revision", "run_config"}
    comparable = {key: value for key, value in actual.items() if key not in ignored}
    matches = [{key: value for key, value in asdict(TrainConfig(**row)).items() if key not in ignored}
               for row in frozen["runs"]]
    if comparable not in matches:
        raise ValueError("Resolved training settings do not match any published run configuration")
    if (ROOT / config.budget_manifest).resolve() != MANIFEST:
        raise ValueError("Stage A must use its single authorized budget manifest")
    if MANIFEST.exists() and remaining_budget(config.budget_manifest) <= 0:
        raise TimeoutError("Stage A absolute deadline has expired")
    receipt = {"source_revision": source, "implementation_tree": git("rev-parse", "HEAD:src/ministrike"),
               "config": relative, "sha256": hashlib.sha256(raw).hexdigest(),
               "registration": config.registration, "checked_utc": datetime.now(timezone.utc).isoformat()}
    destination = ROOT / config.output_dir  # Local: retain full per-execution preflight evidence.
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preflight.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def remaining_budget(path=MANIFEST, start=False, phase=None):
    """Local: enforce an immutable absolute 12h deadline; global: downtime/recovery never refund compute."""
    path = Path(path)  # Local: caller normally supplies the one checked canonical manifest.
    now = datetime.now(timezone.utc)  # Local: absolute time, independent of process uptime.
    if not path.exists():
        if not start:
            raise FileNotFoundError("Stage A budget has not started")
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"version": 1, "experiment": "duel-fixed-v1", "status": "calibration",
                 "started_utc": now.isoformat(), "deadline_utc": (now + timedelta(hours=12)).isoformat(),
                 "source_revision": git("rev-parse", "HEAD"), "budget_seconds": 43200,
                 "active_protocol": "docs/duel-protocol.md", "roadmap": "docs/experiment-roadmap.md",
                 "runs": [], "amendments": [], "legacy_suite": "deferred; never resume automatically"}
        with path.open("x", encoding="utf-8") as stream:  # Local: exclusive creation refuses a second starting clock.
            json.dump(state, stream, indent=2)
    state = read_json(path)
    start_time = datetime.fromisoformat(state["started_utc"])
    deadline = datetime.fromisoformat(state["deadline_utc"])
    if (deadline - start_time).total_seconds() != 43200:
        raise ValueError("Stage A deadline changed from the authorized 12-hour allowance")
    if phase is not None:
        hours = 2 if phase == "A-CAL" or phase.startswith("A-PILOT") else 10  # Local: reserve final evaluation window.
        deadline = min(deadline, start_time + timedelta(hours=hours))  # Local: phases cannot consume later reservations.
    return max(0.0, (deadline - now).total_seconds())  # Local: all later resumes share this same deadline.
