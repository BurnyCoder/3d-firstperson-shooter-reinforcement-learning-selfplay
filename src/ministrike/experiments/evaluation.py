"""Local: frozen-checkpoint tournaments. Global: final-test seeds and results never select training updates.

Source: https://arxiv.org/abs/2108.13264 (preserve uncertainty and disclose the evaluation protocol).
"""

import hashlib  # Local: identify exact checkpoint bytes rather than trusting a reusable filename.
import json  # Local: resume saved raw rounds and their validated protocol.
import logging  # Record every completed evaluation batch without truncating individual round artifacts.
import time  # Enforce the suite deadline even during lengthy tournaments.
from pathlib import Path  # Keep result destinations explicit.

import numpy as np  # Evaluate many independent arenas together.

from ministrike.experiments.analysis import summarize_rounds  # Reuse the pre-specified score/interval implementation.
from ministrike.policies.scripted import RandomPolicy, ScriptedPolicy  # Frozen observation-only baselines.
from ministrike.records import source_revision, write_json  # Save reproducible provenance atomically.
from ministrike.sim import Simulation  # Evaluation uses precisely the same authoritative game.
from ministrike.sim.constants import DT  # Convert fixed ticks into physical elapsed seconds.

LOGGER = logging.getLogger(__name__)  # Runtime configuration decides console/file destinations.
BASELINES = ("random", "rush", "guard")  # Local: one benchmark definition is shared by execution and resume validation.


def policy_identity(policy, identity: str | None, resume: bool) -> str:
    """Local: fingerprint the evaluated policy. Global: resume cannot silently mix different trained actors."""
    if identity is not None:  # Local: generic test/custom policies can supply an explicit stable identity.
        return str(identity)  # Local: callers own the meaning of an explicitly supplied identity.
    checkpoint = getattr(policy, "path", None)  # Local: production PolicyCheckpoint exposes the loaded file path.
    if checkpoint is not None:  # Local: changing bytes at the same path must invalidate resumed evidence.
        with Path(checkpoint).open("rb") as stream:  # Local: hash the file once, without loading another model.
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()  # Local: standard-library streamed digest.
    if resume:  # Local: a class name alone cannot distinguish generic policy parameters.
        raise ValueError("Resuming a policy without a checkpoint path requires an explicit identity")
    return f"unversioned:{type(policy).__module__}.{type(policy).__qualname__}"  # Local: ordinary one-shot generic evaluations remain supported.


def evaluate_policy(policy, output: str | Path, pairs: int = 128, num_envs: int = 16, seed_base: int = 2_000_000, deadline: float | None = None, resume: bool = False, identity: str | None = None) -> dict:
    """Local: run side-swapped rounds against three fixed controls. Global: hold the actor constant throughout evaluation."""
    if pairs < 1 or num_envs < 1:  # Reject nonsensical sample counts before starting game simulation.
        raise ValueError("pairs and num_envs must be positive")  # Explicitly distinguish invalid requests from negative results.
    output = Path(output)  # Results are a single JSON artifact containing all measured rounds.
    protocol = {"version": 1, "policy_identity": policy_identity(policy, identity, resume), "seed_base": seed_base,
                "pairs": pairs, "num_envs": num_envs, "baselines": list(BASELINES)}  # Local: freeze every seed/batching/model decision used for these rounds.
    rows = []  # Keep every round, not only successful examples.
    previous_elapsed = 0.0  # Local: resumed timings include completed earlier evaluation work.
    if resume and output.exists():  # Local: explicit suite resumes retain earlier round evidence.
        previous = json.loads(output.read_text(encoding="utf-8"))  # Local: read the last atomically published artifact.
        if previous.get("protocol") != protocol:  # Local: legacy or differently configured evidence must never mix.
            raise ValueError("Evaluation resume protocol/checkpoint mismatch; use a new output or rerun explicitly")
        rows = previous["rounds"]  # Local: completed records remain unchanged even if their partial batch is replayed.
        previous_elapsed = previous.get("elapsed_seconds", 0.0)  # Local: older interrupted execution consumed real time.
        if previous.get("status") == "complete":  # Local: a complete identical tournament requires no new sampling.
            return previous  # Local: preserve both evidence and timing exactly on an idempotent resume.
    completed = {(row["baseline"], row["pair"], row["side"]) for row in rows}  # Local: one unique record per policy-opponent-role-world combination.
    if len(completed) != len(rows):  # Local: duplicate records would bias score means and confidence intervals.
        raise ValueError("Evaluation resume artifact contains duplicate round identities")
    started = time.monotonic()  # Monotonic durations are immune to clock adjustments.

    def persist(status, reason=None):
        """Local: save complete metadata at every interruption point. Global: running evidence is safely resumable."""
        result = {"status": status, "source_revision": source_revision(), "protocol": protocol,
                  "seed_base": seed_base, "pairs": pairs, "num_envs": num_envs,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                  "rounds": rows, "summary": summarize_rounds(rows)}  # Local: raw outcomes remain the primary evidence.
        if reason is not None:  # Local: incomplete output records why its protocol was not finished.
            result["reason"] = reason  # Local: distinguish a deadline stop from a completed tournament.
        write_json(output, result)  # Local: atomic replacement preserves the last complete JSON record.
        return result  # Local: callers use the same object that was actually persisted.

    for baseline in BASELINES:  # These benchmark identities are frozen in the experiment protocol.
        for offset in range(0, pairs, num_envs):  # Batching amortizes inference while preserving seed-pair identity.
            count = min(num_envs, pairs - offset)  # The final batch never adds unplanned test samples.
            for side in (0, 1):  # Recreate the identical initial arenas with learner roles swapped.
                keys = [(baseline, f"{seed_base + offset}:{arena}", side) for arena in range(count)]  # Local: batch/world identities remain stable across restarts.
                if all(key in completed for key in keys):  # Local: fully completed groups do not repeat expensive matches.
                    continue  # Local: a partially completed group replays only to reconstruct its unsaved game state.
                simulator = Simulation(num_envs=count, seed=seed_base + offset)  # Both sides use the same world seed and arena ordering.
                simulator.reset(seed=seed_base + offset, scenario="standard")  # Test only ordinary 5v5 starts.
                opponent = RandomPolicy(seed_base + offset + 99) if baseline == "random" else ScriptedPolicy(baseline)  # Reinitialize stochastic control consistently.
                learner_slots = slice(side * 5, side * 5 + 5)  # Each learner controls a complete five-player team.
                opponent_slots = slice((1 - side) * 5, (1 - side) * 5 + 5)  # The fixed control owns the opposing team.
                finished = np.zeros(count, dtype=bool)  # Finished arenas remain frozen until this batch completes.
                idle = np.zeros(count)  # Count inactive living decisions, separate from dead-player no-ops.
                living_decisions = np.zeros(count)  # Denominator excludes actions that could not affect the game.
                plants = np.zeros(count, dtype=int)  # Preserve objective events separately from wins.
                defuses = np.zeros(count, dtype=int)  # A team can win by elimination without ever attempting the objective.
                learner_plants = np.zeros(count, dtype=int)  # Local: distinguish the evaluated team's objective behavior.
                learner_defuses = np.zeros(count, dtype=int)  # Local: opponent defuses are not credited to the learner.
                for tick in range(2000):  # One hundred seconds safely covers the documented 60+25-second game limit.
                    if deadline is not None and time.monotonic() >= deadline:  # Respect the shared compute deadline.
                        return persist("incomplete", "deadline")  # Local: preserve every completed round and protocol before stopping.
                    observations = simulator.observe()  # Copies prevent mutable NumPy buffers corrupting stored observations.
                    controls = np.zeros((count, 10, 7), dtype=np.float32)  # Both teams act from the same pre-step state.
                    controls[:, learner_slots] = policy.act(observations[:, learner_slots], deterministic=True)  # Evaluate the actual actor without exploration noise.
                    controls[:, opponent_slots] = opponent.act(observations[:, opponent_slots])  # No private simulator state enters the baseline.
                    living = simulator.alive[:, learner_slots] & (~finished[:, None])  # Ignore finished arenas and dead players.
                    idle += np.sum(living & (np.linalg.norm(controls[:, learner_slots], axis=-1) < 1e-6), axis=1)  # Report literal zero-action inactivity.
                    living_decisions += living.sum(axis=1)  # Keep rates interpretable across varying round lengths.
                    result = simulator.step(controls)  # Physics and objective resolution are shared with play and training.
                    for event in result.events:  # Event records carry their arena index.
                        arena = event.get("arena", 0)  # Backward-compatible default is only relevant to single-arena fixtures.
                        plants[arena] += int(event["type"] == "plant")  # Count objective completions, not reward proxies.
                        defuses[arena] += int(event["type"] == "defuse")  # Keep public event counts independent of policy score.
                        learner_plants[arena] += int(event["type"] == "plant" and event.get("team") == side)  # Local: attribute only the learner's completed plants.
                        learner_defuses[arena] += int(event["type"] == "defuse" and event.get("team") == side)  # Local: attribute only the learner's completed defuses.
                    for arena in np.flatnonzero(result.terminated & (~finished)):  # Write exactly one result for each completed arena.
                        if keys[arena] in completed:  # Local: a partial batch replay must not overwrite or duplicate completed evidence.
                            continue  # Local: its simulation still ran to preserve batched opponent RNG and pairing.
                        winner = int(result.winner[arena])  # -1 is a true game draw, never an evaluation timeout.
                        rows.append({"pair": f"{seed_base + offset}:{arena}", "baseline": baseline, "side": side, "winner": winner,
                                     "score": 0.5 if winner == -1 else float(winner == side), "seconds": (tick + 1) * DT,
                                     "plants": int(plants[arena]), "defuses": int(defuses[arena]),
                                     "learner_plants": int(learner_plants[arena]), "learner_defuses": int(learner_defuses[arena]),
                                     "damage": float(simulator.damage_dealt[arena, side]), "shots": int(simulator.shots_fired[arena, learner_slots].sum()),
                                     "inactive_fraction": float(idle[arena] / max(living_decisions[arena], 1))})  # Preserve exact raw endpoint components.
                        completed.add(keys[arena])  # Local: later resume writes cannot append this round identity twice.
                    finished |= result.terminated  # Do not reset completed worlds or collect extra test games.
                    if finished.all():  # Every world in this side-swapped batch has a terminal result.
                        break  # Continue with the opposite role or next seed batch.
                if not finished.all():  # A broken termination rule must fail visibly rather than inflate draws.
                    raise RuntimeError("Evaluation exceeded the documented game horizon")  # Fix game correctness before interpreting results.
                LOGGER.info("evaluation baseline=%s side=%d seed_batch=%d completed=%d", baseline, side, seed_base + offset, len(rows))  # Live progress remains auditable.
                persist("running")  # Local: every saved intermediate artifact is independently protocol-validatable.
    result = persist("complete")  # Local: mark complete only after all planned unique rounds exist.
    LOGGER.info("evaluation complete %s", result["summary"])  # Summaries never replace the full raw result file.
    return result  # Report generation consumes this same measured result object.
