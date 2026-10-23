"""Local: replay captured native inputs headlessly; global: rendering must not change the authoritative match.

Uses the existing Simulation API and Python JSON: https://docs.python.org/3.12/library/json.html .
This verifies recorded sessions, not human skill or learning quality, and performs no optimization.
"""

import argparse  # Local: explicit checkpoint and evidence paths prevent accidental use of a diagnostic model.
from pathlib import Path  # Local: keep all output under the requested report directory.
import shutil  # Local: preserve original screenshot/replay bytes.

import numpy as np  # Local: restore the exact float32 physical action batches.

from ministrike.experiments.duel_evaluation import file_hash
from ministrike.experiments.registration import read_json
from ministrike.learning.checkpoint import PolicyCheckpoint
from ministrike.records import configure_logging, source_revision, write_json
from ministrike.sim import Simulation


def verify_capture(path, rules):
    """Local: reproduce every recorded transition; global: compare all public canonical fields exactly."""
    record = read_json(path)
    simulation = Simulation(1, seed=record["seed"], experiment=record["experiment"], config=rules)
    round_number = 1
    simulation.reset(seed=record["seed"], scenario="standard")
    for row in record["replay"]:
        if row["round"] != round_number:
            if row["round"] != round_number + 1:
                raise ValueError("Replay skips a round")
            round_number = row["round"]
            simulation.reset(seed=record["seed"] + round_number - 1, scenario="standard")
        simulation.step(np.asarray([row["actions"]], dtype=np.float32))  # Local: no renderer or policy inference is involved.
    if record["round"] == round_number + 1:
        simulation.reset(seed=record["seed"] + record["round"] - 1, scenario="standard")
    if len(record["replay"]) != record["ticks"] or simulation.snapshot() != record["snapshot"]:
        raise ValueError(f"Native/headless mismatch: {path}")
    return {"ticks": record["ticks"], "round": record["round"], "mode": record["mode"],
            "controlled": record["controlled"], "followed": record.get("followed"),
            "paused": record.get("paused"), "done": record["snapshot"]["done"], "exact_parity": True}


def main():
    """Local: load, verify, retain and hash; global: a small orchestration command produces inspectable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("captures", type=Path, nargs="+")
    args = parser.parse_args()
    configure_logging(args.output / "logs")
    policy = PolicyCheckpoint(args.checkpoint, "duel-fixed-v1")
    results = {}
    for path in args.captures:
        row = verify_capture(path, policy.rules)
        for source in (path, path.with_suffix("").with_suffix(".png")):
            destination = args.output / source.name
            shutil.copyfile(source, destination)
            row[source.suffix + "_sha256"] = file_hash(destination)
        results[path.name] = row
    write_json(args.output / "parity.json", {"source_revision": source_revision(),
               "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint), "captures": results})


if __name__ == "__main__":
    main()  # Local: explicit invocation only; importing this verifier never launches a game.
