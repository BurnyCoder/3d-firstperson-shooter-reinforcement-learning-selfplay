"""Local: paired, frozen duel tournaments; global: distinct match and model evidence.

Protocol: docs/duel-protocol.md. Statistical motivation: https://arxiv.org/abs/2108.13264 .
Fixed-layout deterministic matches are counted once per side, never as 128 independent trials.
"""

import hashlib  # Local: bind numerical results to exact checkpoint bytes.
import logging  # Local: persist tournament progress through the shared timestamped handlers.
from pathlib import Path  # Local: evidence remains project-contained by CLI convention.

import numpy as np  # Local: independent per-world random streams and transparent summaries.
import torch  # Local: seeded categorical inference without policy optimization.

from ministrike.learning.checkpoint import PolicyCheckpoint  # Local: inference-only actor, never critic/script targets.
from ministrike.policies.duel import DuelRandomPolicy, DuelScriptedPolicy  # Local: frozen legal benchmarks.
from ministrike.records import write_json, source_revision  # Local: atomic evidence and execution identity.
from ministrike.sim.constants import DT, decode_actions  # Local: canonical action/tick contracts.
from ministrike.sim.simulation import Simulation  # Local: same engine as training/rendering.


def file_hash(path):
    """Local: compute artifact identity; global: numbers cannot ambiguously refer to a mutable model path."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()  # Local: bounded memory for large checkpoints.


def policy_actions(policy, observations, generators, deterministic):
    """Local: batch actor inference with per-world RNG; global: batching does not define the random seed."""
    with torch.no_grad():
        logits = policy.actor(torch.from_numpy(observations.copy()))  # Local: one shared forward call, local inputs only.
        if deterministic:
            categories = logits.argmax(-1)  # Local: same argmax deployment behavior as PolicyCheckpoint.act.
        else:
            probabilities = logits.softmax(-1)  # Local: sample library categorical probabilities with isolated streams.
            categories = torch.stack([torch.multinomial(row.reshape(-1, row.shape[-1]), 1, generator=generator)
                                      .reshape(row.shape[:-1]) for row, generator in zip(probabilities, generators, strict=True)])
        return decode_actions(categories.numpy(), policy.spec.action_nvecs)  # Local: exact shared five-head decoder.


def summarize(rows):
    """Local: balance opponents and spawn ends; global: stochastic counts cannot outweigh deterministic controls."""
    baselines = {}
    for baseline in ("random", "initial", "scripted"):
        group = [row for row in rows if row["baseline"] == baseline]
        sides = [float(np.mean([row["score"] for row in group if row["side"] == side])) for side in (0, 1)]
        baselines[baseline] = {"score": float(np.mean(sides)), "side_scores": sides, "matches": len(group),
                               "wins": sum(row["score"] == 1 for row in group),
                               "draws": sum(row["score"] == .5 for row in group),
                               "losses": sum(row["score"] == 0 for row in group)}
    return {"score": float(np.mean([item["score"] for item in baselines.values()])), "baselines": baselines,
            "elimination_fraction": float(np.mean([row["reason"] == "elimination" for row in rows]))}


def evaluate_duel(checkpoint, initial, output, seed_base=3_000_000, pairs=128, stochastic=False, num_envs=32):
    """Local: evaluate all three controls on paired worlds; global: persist every scheduled outcome and identity."""
    torch.set_num_threads(1)  # Local: tiny inference matrices avoid oversubscribed host thread pools.
    actor = PolicyCheckpoint(checkpoint, "duel-fixed-v1")  # Local: reject incompatible legacy checkpoint dimensions.
    initial_actor = PolicyCheckpoint(initial, "duel-fixed-v1")  # Local: corresponding initialization, not another model.
    identity = {"checkpoint_sha256": file_hash(checkpoint), "initial_sha256": file_hash(initial),
                "seed_base": seed_base, "pairs": pairs, "stochastic": stochastic,
                "source_revision": source_revision(), "experiment": "duel-fixed-v1", "num_envs": num_envs}
    output = Path(output)  # Local: explicit validation/final filenames prevent accidental overwrite.
    if output.exists():
        from ministrike.experiments.registration import read_json
        saved = read_json(output)
        if saved["identity"] != identity:
            raise ValueError("Evaluation output belongs to a different checkpoint or protocol")
        if saved["status"] == "complete":
            return saved  # Local: already completed evidence is not retrained or unnecessarily replayed.
        rows = saved["rounds"]  # Local: completed batch records survive interruption.
    else:
        rows = []
    completed = {(row["baseline"], row["side"], row["seed"]) for row in rows}
    for baseline in ("random", "initial", "scripted"):
        count = pairs if stochastic or baseline == "random" else 1  # Local: deterministic repeats provide no new information.
        for side in (0, 1):
            for offset in range(0, count, num_envs):
                seeds = list(range(seed_base + offset, seed_base + min(count, offset + num_envs)))
                if all((baseline, side, seed) in completed for seed in seeds):
                    continue  # Local: resume preserves prior canonical rows.
                sim = Simulation(len(seeds), experiment="duel-fixed-v1", config=actor.rules)
                obs = sim.observe()  # Local: each world uses identical fixed geometry, distinct policy RNGs.
                randoms = [DuelRandomPolicy(seed) for seed in seeds]
                learner_rng = [torch.Generator().manual_seed(seed + 101) for seed in seeds]
                enemy_rng = [torch.Generator().manual_seed(seed + 202) for seed in seeds]
                counters = [{"hits": 0, "kills": 0, "reload_ticks": 0, "aim_saturated_ticks": 0,
                             "inactive_ticks": 0, "decisions": 0} for _ in seeds]
                reasons = {}  # Local: record actual rule outcomes, never infer draws from a runner timeout.
                for _ in range(400):
                    if sim.done.all():
                        break
                    controls = np.zeros((len(seeds), 2, 7), np.float32)
                    controls[:, side:side + 1] = policy_actions(actor, obs[:, side:side + 1], learner_rng, not stochastic)
                    enemy = 1 - side  # Local: swap roles without changing the paired seed stream.
                    enemy_obs = obs[:, enemy:enemy + 1]
                    if baseline == "random":
                        controls[:, enemy] = np.stack([policy.act(row)[0] for policy, row in zip(randoms, enemy_obs, strict=True)])
                    elif baseline == "initial":
                        controls[:, enemy:enemy + 1] = policy_actions(initial_actor, enemy_obs, enemy_rng, not stochastic)
                    else:
                        controls[:, enemy:enemy + 1] = DuelScriptedPolicy().act(enemy_obs)
                    active = ~sim.done.copy()  # Local: count only genuine preterminal decisions.
                    result = sim.step(controls)
                    obs = result.observations
                    for arena in np.flatnonzero(active):
                        counter = counters[arena]
                        counter["decisions"] += 1
                        counter["reload_ticks"] += int(sim.reload_timer[arena, side] > 0)
                        counter["aim_saturated_ticks"] += int(abs(sim.pitch[arena, side]) >= 1.3)
                        counter["inactive_ticks"] += int(np.linalg.norm(controls[arena, side]) < 1e-6)
                    for event in result.events:
                        arena = event["arena"]
                        counters[arena]["hits"] += int(event["type"] == "damage" and event.get("attacker") == side)
                        counters[arena]["kills"] += int(event["type"] == "kill" and event.get("player") == enemy)
                        if event["type"] == "round_end":
                            reasons[arena] = event["reason"]
                if not sim.done.all():
                    raise RuntimeError("Duel exceeded its registered 400-tick horizon")
                for arena, seed in enumerate(seeds):
                    if (baseline, side, seed) in completed:
                        continue
                    shots = int(sim.shots_fired[arena, side])
                    rows.append({"baseline": baseline, "side": side, "seed": seed,
                                 "score": .5 if sim.winner[arena] < 0 else float(sim.winner[arena] == side),
                                 "winner": int(sim.winner[arena]), "reason": reasons[arena],
                                 "seconds": float(sim.tick[arena] * DT), "damage": float(sim.damage_dealt[arena, side]),
                                 "shots": shots, "accuracy": counters[arena]["hits"] / max(shots, 1), **counters[arena]})
                write_json(output, {"status": "running", "identity": identity, "rounds": rows})
                logging.info("duel_evaluation baseline=%s side=%s rounds=%s", baseline, side, len(rows))
    result = {"status": "complete", "identity": identity, "rounds": rows, "summary": summarize(rows)}
    write_json(output, result)  # Local: summary is derived from the retained raw records.
    return result
