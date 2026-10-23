"""Local: assess a completed registered reward pilot; global: select only from validation evidence.

Engineering gate/selection rule: docs/duel-protocol.md. Paired uncertainty is delegated to
the shared analysis, with its primary statistical sources. No training or final-test access.
"""

import argparse  # Local: require an explicit registered reward identity.
import hashlib  # Local: compare initial actor tensors, not metadata-dependent checkpoint bytes.
from pathlib import Path  # Local: fixed project-local pilot and evidence locations.
import shutil  # Local: publish exact evidence bytes with the standard-library copy operation.

from ministrike.experiments.duel_analysis import paired_match_interval  # Local: preserve evaluation pairing.
from ministrike.experiments.duel_evaluation import file_hash  # Local: checkpoint/file provenance uses one implementation.
from ministrike.experiments.registration import read_json  # Local: reuse record decoding.
from ministrike.learning.checkpoint import load_checkpoint  # Local: restricted checkpoint loading.
from ministrike.records import timestamp, write_json  # Local: durable, timestamped decisions.


def actor_hash(path, component="actor"):
    """Local: hash names, dtypes, dimensions and tensor bytes; global: reward pilots share exact initialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(load_checkpoint(path)[component].items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())  # Local: distinguish shape/name collisions.
        digest.update(tensor.contiguous().numpy().tobytes())  # Local: exact CPU tensor bytes, no rounded decimal strings.
    return digest.hexdigest()


def diagnostic_metrics(evaluation):
    """Local: aggregate recorded counters; global: changing loss alone cannot demonstrate combat learning."""
    metrics = {}
    for baseline in ("random", "initial", "scripted"):
        rows = [row for row in evaluation["rounds"] if row["baseline"] == baseline]
        def total(key, records=rows):
            """Local: pool counters; global: avoid averaging unequal-denominator ratios."""
            return sum(row[key] for row in records)  # Local: bind this baseline's rows at function definition.
        metrics[baseline] = {"damage_per_round": total("damage") / len(rows),
                             "shots_per_round": total("shots") / len(rows),
                             "hits_per_shot": total("hits") / max(1, total("shots")),
                             "reload_occupancy": total("reload_ticks") / total("decisions"),
                             "aim_saturation": total("aim_saturated_ticks") / total("decisions"),
                             "seconds_per_round": total("seconds") / len(rows)}
    return metrics


def publish_evidence(run, reward):
    """Local: copy raw records and hash both endpoints; global: negative pilots are publicly reproducible."""
    destination = Path("reports/duel-fixed-v1/data") / f"pilot-{reward}"
    destination.mkdir(parents=True, exist_ok=True)
    evidence = {}  # Local: hashes address immutable bytes; checkpoint binaries are published separately on HF.
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".log"}:
            shutil.copyfile(path, destination / path.name)
            evidence[path.name] = file_hash(path)
    group = run.parents[1] / f"{reward}.json"  # Local: supervision includes hardware samples and exit status.
    shutil.copyfile(group, destination / "group.json")
    evidence["group.json"] = file_hash(group)
    evidence["checkpoints"] = {name: file_hash(run / name) for name in ("initial.pt", "model.pt")}
    write_json(destination / "evidence.json", evidence)  # Local: no secrets, model tensors or arbitrary files are copied.


def assess(reward):
    """Local: apply the preregistered numerical gate; global: incomplete runs cannot qualify by score alone."""
    root = Path("runs/duel-fixed-v1")
    run = root / "pilots" / reward / "seed5201"
    initial, final = [read_json(run / name) for name in ("initial_validation.json", "final_validation.json")]
    config, result = read_json(run / "config.json"), read_json(run / "result.json")
    if config["seed"] != 5201 or config["reward"] != reward:
        raise ValueError("Pilot seed/reward identity differs from registration")
    if initial["status"] != "complete" or final["status"] != "complete":
        raise ValueError("Both validation endpoints must finish")
    for evaluation in (initial, final):
        if evaluation["identity"]["seed_base"] != 3_000_000 or evaluation["identity"]["stochastic"]:
            raise ValueError("Selection requires registered deterministic validation")
    expected_initial = file_hash(run / "initial.pt")
    if initial["identity"]["checkpoint_sha256"] != expected_initial or any(
            item["identity"]["initial_sha256"] != expected_initial for item in (initial, final)):
        raise ValueError("Corresponding initial checkpoint identity differs")
    if final["identity"]["checkpoint_sha256"] != file_hash(run / "model.pt"):
        raise ValueError("Final validation refers to a different checkpoint")
    fingerprint = actor_hash(run / "initial.pt")
    reference = actor_hash(root / "pilots/R0/seed5201/initial.pt")
    if fingerprint != reference:
        raise ValueError("Reward candidates did not start with identical actor parameters")
    critic = actor_hash(run / "initial.pt", "critic")  # Local: equal initialization includes the separate value network.
    if critic != actor_hash(root / "pilots/R0/seed5201/initial.pt", "critic"):
        raise ValueError("Reward candidates did not start with identical critic parameters")
    delta = final["summary"]["score"] - initial["summary"]["score"]
    target = read_json("reports/duel-fixed-v1/calibration/selection.json")["pilot_arena_steps"]
    complete = result["complete"] and result["frames"] == config["total_frames"] == target
    assessment = {"reward": reward, "seed": config["seed"], "assessed_utc": timestamp(),
                  "source_revision": result["source_revision"], "initial_actor_sha256": fingerprint,
                  "initial_critic_sha256": critic,
                  "initial_checkpoint_sha256": expected_initial, "final_checkpoint_sha256": file_hash(run / "model.pt"),
                  "complete_equal_budget": complete, "initial": initial["summary"], "final": final["summary"],
                  "improvement": delta, "numerical_gate_passed": complete and delta >= .10,
                  "match_uncertainty": paired_match_interval(initial, final),
                  "diagnostics": {"initial": diagnostic_metrics(initial), "final": diagnostic_metrics(final)},
                  "scope": "Exploratory seed5201 selection; correctness review remains a separate prerequisite"}
    write_json(run / "assessment.json", assessment)
    publish_evidence(run, reward)  # Local: publishing the next reward's prerequisite includes all failed evidence.
    print(assessment)  # Local: full structured result is also saved without truncation.
    return assessment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reward", choices=("R0", "R1", "R2"))
    assess(parser.parse_args().reward)  # Local: explicit invocation only; importing never selects a model.
