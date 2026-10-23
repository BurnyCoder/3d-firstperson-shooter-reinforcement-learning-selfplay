"""Local: retain each incomplete 5v5 run; global: archival reports never invent a completed tournament.

File copying and JSON use Python 3.12: https://docs.python.org/3.12/library/shutil.html .
The original protocol, checkpoint provenance and budget remain unchanged.
"""

from pathlib import Path  # Local: fixed project paths delimit this archival operation.
import json  # Local: inspect the last complete canonical metric record without altering its bytes.
import shutil  # Local: preserve exact public evidence bytes instead of rewriting metrics.

from ministrike.experiments.duel_evaluation import file_hash  # Local: reuse the checkpoint/evidence hash implementation.
from ministrike.experiments.registration import read_json  # Local: reuse strict JSON decoding.
from ministrike.records import configure_logging, timestamp, write_json  # Local: shared timestamped evidence output.


def main():
    """Local: copy, summarize and hash six archived endpoints; global: no training or evaluation is executed."""
    destination = Path("reports/deferred-5v5")  # Local: keep derived reporting separate from original runs.
    configure_logging(destination / "logs")
    records = []  # Local: one record per condition/seed, including unsuccessful models.
    for stage in sorted(Path("runs/suite/publication").glob("*_seed*")):
        archive = read_json(stage / "archive-status.json")  # Local: publication already checked the metric/checkpoint boundary.
        target = destination / "data" / stage.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("archive-status.json", "config.json", "metrics.jsonl"):
            shutil.copyfile(stage / name, target / name)  # Local: retain original numbers and exact hashes.
        last = json.loads((target / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        if archive["complete"] or last["frames"] != archive["frames"] or last["updates"] != archive["updates"]:
            raise ValueError("Deferred checkpoint/metric boundary mismatch")  # Local: fail rather than describe mixed evidence.
        name = stage.name
        report = (f"# Deferred 5v5: {name}\n\n**Incomplete, not a deployment recommendation.** "
                  f"Seed {archive['seed']}, condition `{archive['condition']}`, completed {archive['frames']:,} "
                  f"of 4,167,680 arena steps ({archive['updates']:,} updates) before the user deferred this experiment. "
                  "No deadline or target was extended and no final tournament was completed.\n\n"
                  f"The last canonical training record reports {last['rounds']:,} rounds, {last['wins']:,} wins "
                  f"and {last['draws']:,} draws. These are changing-opponent training outcomes, not fixed-benchmark scores "
                  "or evidence of superiority over the other condition. Unequal completed exposure prevents the registered "
                  "equal-budget comparison.\n\n"
                  f"[Original config](data/{name}/config.json), [canonical metrics](data/{name}/metrics.jsonl), "
                  f"[derived archive status and both checkpoint hashes](data/{name}/archive-status.json). "
                  f"Initial source identity: `{archive['source_revision']}`. The archive's checkpoint elapsed time is "
                  f"{archive['elapsed_seconds']:.6f} seconds; [the suite ledger](data/suite.json) additionally preserves "
                  "interruption/recovery accounting. A final metric timestamp can follow checkpoint serialization.\n\n"
                  "The [original protocol](../../docs/protocol.md) defines the tactical map, 5v5 rules, two-layer "
                  "128-unit shared actor, two-layer 256-unit centralized critic, MAPPO/PPO, curriculum, opponents "
                  "and shaped reward. Fixed-condition training uses the historical scripted controller; this is "
                  "outside the later self-play-only Stage A protocol.\n\n"
                  "[Learning audit](../learning-audit/summary.md), [hardware probes and interruptions](../performance/summary.md), "
                  "[loss graphs](../learning-audit/losses/README.md), [aggregate limitations](README.md). "
                  f"[Public checkpoint pair](https://huggingface.co/BurnyCoder/ministrike-{name.replace('_', '-')}).\n")
        (destination / f"{name}.md").write_text(report, encoding="utf-8")
        records.append({"run": name, "archive": archive, "last_training_record": last})
    shutil.copyfile("runs/suite/suite.json", destination / "data/suite.json")  # Local: preserve the original deferral/deadline ledger.
    write_json(destination / "audit.json", {"created_utc": timestamp(), "runs": records,
               "raw_sha256": {p.relative_to(destination).as_posix(): file_hash(p) for p in (destination / "data").rglob("*") if p.is_file()}})


if __name__ == "__main__":
    main()  # Local: explicit reporting invocation only; imports perform no archival mutation.
