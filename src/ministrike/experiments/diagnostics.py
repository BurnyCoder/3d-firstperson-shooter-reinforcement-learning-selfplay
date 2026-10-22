"""Local: frozen-policy behavior audits; global: diagnose learning without final-test leakage.

These exploratory matches use validation seeds and never update weights or primary results.
Sources: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html
and https://docs.python.org/3/library/logging.html .
"""

import argparse  # Local: explicit snapshot/output choices; global: reproducible diagnostic invocations.
import logging  # Local: stream every completed probe; global: preserve failures and negative findings.
from pathlib import Path  # Local: group evidence beside its frozen checkpoint copies.

import numpy as np  # Local: vectorized counters use the same arena/player axes as the simulator.
import torch  # Local: seed stochastic action sampling without touching another training process.

from ministrike.learning.checkpoint import PolicyCheckpoint  # Reuse actual gameplay inference.
from ministrike.policies.scripted import ScriptedPolicy  # Reuse the exact fixed training opponent.
from ministrike.records import configure_logging, source_revision, timestamp, write_json
from ministrike.sim.constants import DT  # Local: diagnostic durations use authoritative simulation ticks.
from ministrike.sim.simulation import Simulation  # Global: no simplified diagnostic physics.

LOGGER = logging.getLogger(__name__)  # Local: the CLI installs full timestamped file/terminal handlers.


def behavior_probe(policy, scenario, seed, pairs=16, deterministic=False):
    """Local: inspect complete side-swapped rounds; global: score actual behavior, not loss magnitude."""
    if pairs < 1:  # Local: an empty comparison cannot supply a meaningful score.
        raise ValueError("pairs must be positive")
    rows = []  # Local: retain every completed match rather than only aggregate successes.
    for side in (0, 1):  # Local: both roles see identical initialized worlds and the same baseline.
        torch.manual_seed(seed + side)  # Local: common action RNG initialization for compared checkpoints.
        sim = Simulation(pairs, seed=seed)  # Local: arena index is part of reproducible seed identity.
        observations = sim.reset(seed=seed, scenario=scenario)  # Local: select an explicit diagnostic curriculum.
        opponent = ScriptedPolicy()  # Local: balanced is the original fixed training control.
        slots = slice(side * 5, side * 5 + 5)  # Local: all five learner slots retain the same role this round.
        others = slice((1 - side) * 5, (1 - side) * 5 + 5)  # Local: complementary opponent slots.
        finished = np.zeros(pairs, dtype=bool)  # Local: each arena contributes exactly one outcome.
        counters = np.zeros((pairs, 5), dtype=float)  # Local: living decisions, fire intent, reload intent, reloading, steep aim.
        interaction = np.zeros(pairs)  # Local: measure how far held objective attempts progress.
        plants = np.zeros(pairs, dtype=int)  # Local: credit only the learner's successful plants.
        defuses = np.zeros(pairs, dtype=int)  # Local: credit only the learner's successful defuses.
        reasons = {}  # Local: terminal event reasons distinguish combat from objective or timeout losses.
        for tick in range(1701):  # Local: the full 60-second round plus 25-second bomb fits this horizon.
            controls = np.zeros((pairs, 10, 7), dtype=np.float32)  # Local: both teams act from one pre-step state.
            controls[:, slots] = policy.act(observations[:, slots], deterministic=deterministic)
            controls[:, others] = opponent.act(observations[:, others])  # Local: no private state enters the control.
            alive = sim.alive[:, slots] & ~finished[:, None]  # Local: exclude impossible dead/post-round decisions.
            measures = (np.ones_like(alive), controls[:, slots, 4] > .5, controls[:, slots, 5] > .5,
                        sim.reload_timer[:, slots] > 0, np.abs(sim.pitch[:, slots]) > np.pi / 3)
            counters += np.stack([(item & alive).sum(-1) for item in measures], axis=-1)  # Local: common denominator.
            result = sim.step(controls)  # Global: gameplay and diagnostic transitions are identical.
            observations = result.observations  # Local: consume resolved observations without an extra simulation step.
            interaction = np.maximum(interaction, sim.interaction[:, slots].max(-1))  # Local: keep longest held attempt.
            for event in result.events:  # Local: event attribution is read before any possible reset.
                arena = event["arena"]  # Local: events carry an unambiguous independent-world index.
                plants[arena] += int(event["type"] == "plant" and event.get("team") == side)
                defuses[arena] += int(event["type"] == "defuse" and event.get("team") == side)
                if event["type"] == "round_end":  # Local: preserve the actual rule that ended this match.
                    reasons[arena] = event["reason"]  # Local: completed arenas cannot emit another terminal event.
            for arena in np.flatnonzero(result.terminated & ~finished):  # Local: count each terminal state once.
                winner = int(result.winner[arena])  # Local: zero/one are roles and minus one is a real draw.
                count = max(counters[arena, 0], 1)  # Local: prevent division by zero for diagnostic edge fixtures.
                rows.append({"seed": seed, "arena": int(arena), "side": side, "winner": winner,
                             "score": .5 if winner < 0 else float(winner == side), "reason": reasons[arena],
                             "seconds": (tick + 1) * DT, "damage": float(sim.damage_dealt[arena, side]),
                             "shots": int(sim.shots_fired[arena, slots].sum()), "plants": int(plants[arena]),
                             "defuses": int(defuses[arena]), "max_interaction_seconds": float(interaction[arena]),
                             **{key: float(value / count) for key, value in zip(
                                 ("fire_intent", "reload_intent", "reloading_fraction", "steep_aim_fraction"),
                                 counters[arena, 1:], strict=True)}})  # Local: store raw rates with named physical meanings.
            finished |= result.terminated  # Local: freeze completed arenas while longer matches finish.
            if finished.all():  # Local: no unnecessary padding after the last real outcome.
                break  # Local: advance to the other learner side.
        if not finished.all():  # Global: broken termination is an error, never a fabricated draw.
            raise RuntimeError("Diagnostic round exceeded the game horizon")
    return rows  # Local: callers retain all matches and choose explicit, transparent summaries.


def audit_snapshots(root, pairs=16):
    """Local: compare frozen initial/current actors and a positive control; global: preserve all probe outcomes."""
    root = Path(root)  # Local: the caller must freeze files before this entry point is invoked.
    if (root / "behavior-audit.json").exists():  # Global: repeated diagnostics must not overwrite earlier evidence.
        raise FileExistsError("Use a fresh snapshot directory for another behavior audit")
    policies = {"initial": PolicyCheckpoint(root / "fixed_seed111" / "initial.pt"),
                "fixed_current": PolicyCheckpoint(root / "fixed_seed111" / "model.pt"),
                "selfplay_current": PolicyCheckpoint(root / "selfplay_seed111" / "model.pt"),
                "scripted_control": ScriptedPolicy()}  # Local: identical scripted teams test scoring/side symmetry.
    result = {"timestamp_utc": timestamp(), "source_revision": source_revision(), "exploratory": True,
              "pairs": pairs, "deterministic": False, "opponent": "balanced", "probes": []}
    for scenario in ("combat", "objective", "standard"):  # Local: retain each curriculum separately.
        for name, policy in policies.items():  # Local: no model selection or hidden discard rule.
            rows = behavior_probe(policy, scenario, 1_100_010, pairs)  # Global: keep final-test root 2,000,000 untouched.
            summary = {key: float(np.mean([row[key] for row in rows])) for key in (
                "score", "damage", "shots", "seconds", "plants", "defuses", "max_interaction_seconds",
                "fire_intent", "reload_intent", "reloading_fraction", "steep_aim_fraction")}
            result["probes"].append({"policy": name, "scenario": scenario, "summary": summary, "rounds": rows})
            write_json(root / "behavior-audit.json", result)  # Local: save progress after every complete comparison cell.
            LOGGER.info("behavior_probe policy=%s scenario=%s summary=%s", name, scenario, summary)
    return result  # Local: complete raw evidence is available to the report phase.


def main():
    """Local: parse, configure, audit; global: a readable orchestration wrapper with no training side effects."""
    parser = argparse.ArgumentParser(description=__doc__)  # Local: the command documents its exploratory scope.
    parser.add_argument("--snapshot", type=Path, required=True)  # Local: never inspect a concurrently replaced model.
    parser.add_argument("--pairs", type=int, default=16)  # Local: an explicit equal match budget applies to every policy.
    args = parser.parse_args()  # Local: reject unknown options before evaluating anything.
    configure_logging(args.snapshot / "logs")  # Global: full diagnostic outputs persist beside their raw evidence.
    torch.set_num_threads(1)  # Local: one diagnostic process does not oversubscribe the active six learners.
    audit_snapshots(args.snapshot, args.pairs)  # Local: complete the frozen policy comparisons.


if __name__ == "__main__":  # Local: imports remain side-effect free for tests and reporting.
    main()  # Local: run only the requested diagnostic command.
