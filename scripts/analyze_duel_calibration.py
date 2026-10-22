"""Local: resolve the registered hardware comparison; global: publish decisions before pilots.

Selection is an engineering rule in docs/duel-protocol.md, not an optimality claim.
JSON/path APIs: https://docs.python.org/3.12/library/json.html and /pathlib.html .
Run: uv run python scripts/analyze_duel_calibration.py . No optimizer work occurs here.
"""

import math  # Local: finite-value validation and whole-rollout rounding.
from pathlib import Path  # Local: only this project's saved calibration artifacts are read.
import shutil  # Local: retain exact raw records instead of reconstructing measured evidence.

from ministrike.experiments.duel_evaluation import file_hash  # Local: shared SHA256 provenance.
from ministrike.experiments.registration import read_json  # Local: reuse UTF-8 record parsing.
from ministrike.records import write_json  # Local: reuse atomic evidence publication.


def collect_cells(root, destination):
    """Local: validate complete cells and preserve raw bytes; global: failures remain visible."""
    rows, hashes = [], {}  # Local: separate measurements from provenance metadata.
    for path in sorted((root / "calibration").glob("*.json")):
        cell = read_json(path)  # Local: launcher timing includes collection, updates and final saves.
        telemetry = cell["telemetry"]  # Local: host headroom includes other applications on this machine.
        finite = True  # Local: any nonfinite optimizer metric excludes that cell.
        for worker in cell["workers"]:
            directory = Path(worker["output_dir"])  # Local: registered output location, not an inferred PID.
            for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                import json  # Local: one complete JSON object is stored per update.
                finite &= all(math.isfinite(v) for v in json.loads(line).values() if isinstance(v, (int, float)))
            target = destination / "raw" / directory.relative_to(root)  # Local: retain each independent seed.
            target.mkdir(parents=True, exist_ok=True)
            for raw in directory.iterdir():
                if raw.is_file() and raw.suffix in {".json", ".jsonl", ".log"}:
                    shutil.copyfile(raw, target / raw.name)  # Local: metrics/config/logs are safe public evidence.
                    hashes[str((target / raw.name).relative_to(destination))] = file_hash(raw)
            for label in ("initial.pt", "model.pt"):
                hashes[str((target / label).relative_to(destination))] = file_hash(directory / label)
        target = destination / "raw" / path.name  # Local: group telemetry is not in per-worker reports.
        shutil.copyfile(path, target)
        hashes[str(target.relative_to(destination))] = file_hash(path)
        first = read_json(Path(cell["workers"][0]["output_dir"]) / "config.json")
        ram = min(row["ram_available_fraction"] for row in telemetry)
        vram = min(row.get("gpu_available_fraction", 0) for row in telemetry)
        complete = cell["status"] == "complete" and all(row["complete"] for row in cell["results"])
        rows.append({"name": path.stem, "aggregate_rate": cell["aggregate_arena_steps_per_second"],
                     "wall_seconds": cell["wall_seconds"], "workers": len(cell["workers"]),
                     "num_envs": first["num_envs"], "device": first["device"], "inference_device": first["inference_device"],
                     "ram_headroom": ram, "vram_headroom": vram, "finite": finite, "complete": complete,
                     "eligible": complete and finite and min(ram, vram) >= .20,
                     "source_revision": first["source_revision"]})  # Local: eligibility precedes throughput ranking.
    return rows, hashes


def main():
    """Local: select then write a reproducible resolution; global: never start a dependent experiment."""
    root = Path("runs/duel-fixed-v1")
    destination = Path("reports/duel-fixed-v1/calibration")
    destination.mkdir(parents=True, exist_ok=True)
    rows, hashes = collect_cells(root, destination)
    eligible = [row for row in rows if row["eligible"]]  # Local: reject low-headroom cells even if fastest.
    fastest = max(row["aggregate_rate"] for row in eligible)
    tied = [row for row in eligible if row["aggregate_rate"] >= .95 * fastest]
    selected = min(tied, key=lambda row: (1 - row["vram_headroom"], 1 - row["ram_headroom"], row["workers"]))
    batch = selected["num_envs"] * 128  # Local: pilot stops at a complete registered rollout boundary.
    target = max(batch, math.floor(.8 * selected["aggregate_rate"] / selected["workers"] * 1200 / batch) * batch)
    write_json(destination / "selection.json", {"cells": rows, "selected": selected,
               "pilot_arena_steps": target, "pilot_max_seconds": 1200,
               "tie_rule": "Within 5%, lower observed VRAM then host memory; fewer workers if equal",
               "budget": read_json(root / "manifest.json")})  # Local: snapshot the original absolute clock.
    write_json(destination / "evidence.json", hashes)
    print({"selected": selected, "pilot_arena_steps": target})  # Local: concise terminal receipt; raw evidence stays complete.


if __name__ == "__main__":
    main()  # Local: importing this analysis does not mutate evidence or execute experiments.
