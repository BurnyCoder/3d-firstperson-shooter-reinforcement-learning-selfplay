"""Local: a simple real-physics learning control; global: separate optimizer failure from task difficulty.

Only this diagnostic harness replaces starting states and opposition. Production training is untouched.
The actual simulator, categorical actor, GAE, PPO, optimizer, and checkpoint path remain in use.
Motivation: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
Fixture isolation: https://docs.python.org/3/library/unittest.mock.html#patch
"""

import argparse  # Local: explicit experimental output and seed choices.
import logging  # Local: retain complete timestamped measurements and errors.
from pathlib import Path  # Local: isolated artifacts must never overwrite primary runs.
from unittest.mock import patch  # Local: replace the environment only within this test harness's process.

import numpy as np  # Local: starting-state interventions preserve batched arena axes.
import torch  # Local: control inference RNG and thread count independently of live learners.

from ministrike.learning.checkpoint import PolicyCheckpoint  # Local: evaluate saved weights through real inference.
from ministrike.learning.config import TrainConfig  # Local: retain original model/PPO hyperparameters.
from ministrike.learning.environment import ArenaEnv  # Local: use the production TorchRL bridge.
from ministrike.learning.models import ActorNetwork  # Local: the initial pool uses the real actor architecture.
from ministrike.learning.opponents import SnapshotPool  # Local: preserve the environment's interface without scripted aiming.
from ministrike.learning.trainer import train_run  # Global: run precisely the production optimizer/collection orchestration.
from ministrike.records import configure_logging, source_revision, timestamp, write_json
from ministrike.sim.constants import GameConfig  # Local: shorten only the diagnostic episode deadline.
from ministrike.sim.observation import make_frame  # Local: reset history reflects intervened starting states.
from ministrike.sim.simulation import Simulation  # Global: shooting and hit geometry remain authoritative.

LOGGER = logging.getLogger(__name__)  # Local: diagnostics share project log formatting.


class PassivePolicy:
    """Local: an inactive target; global: remove opponent difficulty while retaining physical hit detection."""

    def act(self, observation):
        """Local: return physical no-op controls; global: the learner still chooses all its own six controls."""
        return np.zeros((*observation.shape[:-1], 7), dtype=np.float32)  # Local: no target behavior depends on hidden state.


class FiringRangeEnv(ArenaEnv):
    """Local: one attacker shoots a one-hit target within four ticks; global: a positive learning control only."""

    def __init__(self, num_envs, seed, condition, pool, scenario="combat"):
        """Local: install a short diagnostic arena; global: do not modify production GameConfig defaults."""
        super().__init__(num_envs, seed, "fixed", pool, scenario)  # Local: preserve the same learner/opponent data interface.
        self.sim = Simulation(num_envs, seed, GameConfig(round_seconds=.2))  # Local: failure has a rapid terminal signal.
        self.fixed_policy = PassivePolicy()  # Local: this diagnostic removes return fire rather than assisting the learner.

    def _reset(self, tensordict=None, **kwargs):
        """Local: create an observable shooting opportunity; global: partial resets still preserve other arenas."""
        super()._reset(tensordict, **kwargs)  # Local: retain seeded round/reset/history machinery.
        requested = None if tensordict is None else tensordict.get("_reset", None)  # Local: inspect the exact library mask.
        mask = np.ones(self.batch_size[0], dtype=bool) if requested is None else requested.numpy().reshape(-1)
        self.roles[mask] = 0  # Local: attacker success requires shooting; a timeout cannot grant a free victory.
        self.sim.health[mask] = 0  # Local: unused fixed slots remain present but cannot act or block bullets.
        self.sim.health[mask, 0] = 100  # Local: one learner is alive in its ordinary attacker slot.
        self.sim.health[mask, 5] = 25  # Local: one real rifle hit is sufficient to eliminate the target.
        self.sim.positions[mask, 0] = [-8, 0, -5.5]  # Local: use the map's unobstructed combat lane.
        self.sim.positions[mask, 5] = [-8, 0, -3]  # Local: the target starts 2.5 metres ahead.
        self.sim.positions[mask, 5, 0] += self.sim.rng.uniform(-.4, .4, int(mask.sum()))  # Local: aiming still depends on observations.
        self.sim.yaw[mask, 0] = 0  # Local: initial orientation is a starting-state choice, never policy assistance.
        self.sim.pitch[mask, 0] = 0  # Local: the actor can freely change pitch on every decision.
        self.sim.bomb_planted[mask] = False  # Local: all curriculum phases in this harness remain the same shooting task.
        self.sim.bomb_carrier[mask] = 0  # Local: the ordinary attack/defense terminal rules remain valid.
        self.sim.bomb_position[mask] = self.sim.positions[mask, 0]  # Local: no stale objective position leaks from reset.
        self.sim.history[mask] = make_frame(self.sim)[mask, :, None]  # Local: all history slots contain truthful initial features.
        return self.current_tensordict()  # Local: the actor sees the same observation schema as the full game.


def score_firing_range(policy, seed, rounds=512):
    """Local: score fixed disjoint starting worlds; global: improved training loss alone cannot pass the control."""
    if rounds < 16 or rounds % 16:  # Local: the declared fixed-size batches must cover every requested world.
        raise ValueError("rounds must be a positive multiple of 16")
    torch.manual_seed(seed)  # Local: pair the initial RNG stream while allowing policies to choose different actions.
    env = FiringRangeEnv(16, seed, "fixed", SnapshotPool(ActorNetwork(), seed))  # Local: identical world sequence per checkpoint.
    scores = []  # Local: preserve each raw binary shooting success.
    for _ in range(rounds // 16):  # Local: a fixed world count avoids selecting easier early-ending matches.
        env.reset()  # Local: all sixteen worlds start a new independent attempt.
        observation = env.sim.observe()  # Local: inference receives the actual visibility-filtered actor history.
        for _ in range(4):  # Local: no round lasts longer than the four-tick deadline.
            controls = np.zeros((16, 10, 7), dtype=np.float32)  # Local: passive opposition remains physically inactive.
            controls[:, :5] = policy.act(observation[:, :5], deterministic=False)  # Local: evaluate stochastic training behavior.
            observation = env.sim.step(controls).observations  # Local: terminal worlds freeze under the normal simulator rules.
        assert env.sim.done.all()  # Local: a missing result is an implementation error, not a failed shot.
        scores.extend((env.sim.winner == 0).astype(int).tolist())  # Local: attacker wins exactly when the target is eliminated.
    env.close()  # Local: use the normal environment lifecycle.
    return {"rounds": len(scores), "score": float(np.mean(scores)), "outcomes": scores}  # Global: retain negative outcomes too.


def run_probe(output, seed, device):
    """Local: execute a fixed-budget positive control; global: never promote this toy checkpoint as a trained game bot."""
    output = Path(output)  # Local: the run identity belongs to the diagnostic namespace.
    config = TrainConfig(output_dir=str(output), seed=seed, condition="fixed", device=device,
                         total_frames=65_536, max_seconds=600, torch_threads=1)  # Local: exactly 32 production updates.
    configure_logging(output / "logs")  # Local: file and terminal retain the full run and any exceptions.
    write_json(output / "probe-protocol.json", {"kind": "exploratory_positive_control", "seed": seed,
               "source_revision": source_revision(), "timestamp_utc": timestamp(), "frames": config.total_frames,
               "validation_seed": 1_190_000, "success": "Final stochastic score >= 0.8 and improvement >= 0.2 over initial on 512 fixed worlds.",
               "limitations": "A four-tick shooting fixture does not establish 5v5 skill or self-play superiority."})
    with patch("ministrike.learning.trainer.ArenaEnv", FiringRangeEnv):  # Local: isolate the fixture to this dedicated process.
        trained = train_run(config)  # Global: all actual collection, loss, synchronization and saving code is reused.
    initial = score_firing_range(PolicyCheckpoint(output / "initial.pt"), 1_190_000)  # Local: evaluate untouched initial weights.
    final = score_firing_range(PolicyCheckpoint(output / "model.pt"), 1_190_000)  # Local: pair the exact same validation worlds.
    result = {"training": trained, "initial": initial, "final": final,
              "passed": final["score"] >= .8 and final["score"] - initial["score"] >= .2}
    write_json(output / "probe-result.json", result)  # Local: an unsuccessful control remains a first-class result.
    LOGGER.info("probe_result initial=%s final=%s passed=%s", initial["score"], final["score"], result["passed"])
    return result  # Local: analysis consumes recorded outcomes rather than console impressions.


def main():
    """Local: parse and run the isolated control; global: primary suite execution remains independent."""
    parser = argparse.ArgumentParser(description=__doc__)  # Local: document the diagnostic's restricted scope.
    parser.add_argument("--output", type=Path, required=True)  # Local: require an explicit fresh evidence destination.
    parser.add_argument("--seed", type=int, required=True)  # Local: never silently choose the best-looking random seed.
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")  # Local: reuse verified local acceleration.
    args = parser.parse_args()  # Local: reject unsupported settings before training.
    torch.set_num_threads(1)  # Local: preserve CPU capacity for the six live experiments.
    run_probe(args.output, args.seed, args.device)  # Local: run the specified fixed-budget control once.


if __name__ == "__main__":  # Local: importing fixtures for regression tests does not train anything.
    main()  # Local: explicitly requested diagnostic CLI execution.
