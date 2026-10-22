"""Local: frozen experiment bookkeeping regressions. Global: incomplete compute must never become completed evidence."""

from datetime import UTC, datetime, timedelta  # Local: give fixtures a valid shared deadline without running training.
from types import SimpleNamespace  # Local: minimal transition fixtures isolate orchestration from physics.

import numpy as np  # Local: preserve real batched shapes in fast evaluation fixtures.
import pytest  # Local: assert mismatched evidence cannot be resumed.

from ministrike.experiments import evaluation  # Local: monkeypatch only the expensive simulation boundary.
from ministrike.experiments.suite import RUN_ORDER, run_suite  # Local: exercise the actual resume/completion wrapper.
from ministrike.records import write_json  # Local: use the production atomic evidence format.
from ministrike.sim.constants import OBS_DIM  # Local: fixture policies receive the actual observation contract.


def test_suite_does_not_mark_unequal_frame_runs_complete(tmp_path):
    """Local: one timed-out run invalidates full completion. Global: six artifacts do not imply six equal-budget experiments."""
    runs = {}  # Local: construct recorded results without optimizing any policy.
    for index, (condition, seed) in enumerate(RUN_ORDER):  # Local: include every pre-specified experimental identity.
        name = f"{condition}_seed{seed}"  # Local: production directory naming is part of the resume contract.
        runs[name] = {"complete": index != 0, "frames": 128 if index else 64}  # Local: one run missed its frozen target.
        write_json(tmp_path / name / "result.json", runs[name])  # Local: resume must retain the time-limited result.
        write_json(tmp_path / name / "evaluation.json", {"status": "complete"})  # Local: evaluation itself finished.
        write_json(tmp_path / name / "initial_evaluation.json", {"status": "complete"})  # Local: both baseline endpoints exist.
    manifest = {"output_dir": str(tmp_path), "deadline_utc": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                "num_envs": 1, "evaluation_pairs": 1, "total_frames": 128, "per_run_seconds": 30,
                "device": "cpu", "runs": runs, "test_seed_base": 2_000_000}  # Local: bypass calibration using a frozen protocol.
    manifest_path = tmp_path / "suite.json"  # Local: this is the explicit resume entry point.
    write_json(manifest_path, manifest)  # Local: persist the pre-existing suite before invoking production code.
    result = run_suite(tmp_path, resume=manifest_path)  # Local: no learning or evaluation should be repeated.
    assert result["status"] == "budget_exhausted"  # Local: completed tournaments cannot conceal incomplete training.
    assert result["runs"]["selfplay_seed111"]["frames"] == 64  # Local: the partial experimental outcome remains unchanged.


class TwoTickSimulation:
    """Local: one arena finishes per tick. Global: expose partial-batch resume without running real matches."""

    constructions = 0  # Local: count whether already complete batches are unnecessarily replayed.

    def __init__(self, num_envs, seed):
        """Local: allocate exact public metric shapes. Global: evaluation depends only on the simulator contract."""
        type(self).constructions += 1  # Local: test idempotent completed resume without inspecting implementation branches.
        self.num_envs = num_envs  # Local: two arenas produce a genuinely partial first batch.
        self.alive = np.ones((num_envs, 10), dtype=bool)  # Local: learner inactivity denominators remain valid.
        self.damage_dealt = np.zeros((num_envs, 2))  # Local: no damage is needed to test bookkeeping.
        self.shots_fired = np.zeros((num_envs, 10))  # Local: metric interface matches production simulation.
        self.ticks = 0  # Local: first arena terminates at tick one and second at tick two.

    def reset(self, **kwargs):
        """Local: satisfy seeded reset interface. Global: fixture timing remains deterministic."""
        self.ticks = 0  # Local: reset begins the prescribed two-tick trajectory.
        return self.observe()  # Local: match the simulator's canonical reset-observation return contract.

    def observe(self):
        """Local: return legal actor input dimensions. Global: zero policies do not inspect hidden simulator state."""
        return np.zeros((self.num_envs, 10, OBS_DIM), dtype=np.float32)  # Local: observation values do not affect the fixture.

    def step(self, controls):
        """Local: generate one completed arena and objective event per tick. Global: round identities remain stable."""
        self.ticks += 1  # Local: advance each active evaluation fixture deterministically.
        arena = self.ticks - 1  # Local: match fixed array slot to its unique terminal transition.
        events = [{"arena": arena, "type": "plant", "team": 0}] if arena < self.num_envs else []  # Local: preserve explicit team attribution.
        return SimpleNamespace(terminated=np.arange(self.num_envs) < self.ticks,
                               winner=np.zeros(self.num_envs, dtype=int), events=events,
                               observations=self.observe())  # Local: attack wins and canonical post-step observations are deterministic.


class ZeroPolicy:
    """Local: deterministic no-op policy. Global: evaluation resume tests require no neural optimization."""

    def act(self, observation, deterministic=True):
        """Local: preserve arbitrary player batch axes. Global: same public policy API as checkpoint inference."""
        return np.zeros((*observation.shape[:-1], 7), dtype=np.float32)  # Local: legal no-op physical controls.


def test_evaluation_resumes_partial_batch_without_losing_or_duplicating_rounds(tmp_path, monkeypatch):
    """Local: resume reconstructs an unfinished batch. Global: completed evidence survives deadline interruptions."""
    monkeypatch.setattr(evaluation, "Simulation", TwoTickSimulation)  # Local: avoid real game runtime in bookkeeping tests.
    monkeypatch.setattr(evaluation, "ScriptedPolicy", lambda style: ZeroPolicy())  # Local: baseline controls do not affect fixture outcomes.
    times = iter([0.0, 0.0, 2.0])  # Local: deadline arrives after first arena finishes, before the second tick.
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: next(times, 2.0))  # Local: deterministic interruption timing.
    output = tmp_path / "evaluation.json"  # Local: all partial/final writes target the same evidence file.
    partial = evaluation.evaluate_policy(ZeroPolicy(), output, pairs=2, num_envs=2, deadline=1, identity="test-policy")
    assert partial["status"] == "incomplete" and len(partial["rounds"]) == 1  # Local: establish a real partially finished batch.
    original = partial["rounds"][0].copy()  # Local: this completed record must survive replay unchanged.
    complete = evaluation.evaluate_policy(ZeroPolicy(), output, pairs=2, num_envs=2, resume=True, identity="test-policy")
    assert complete["status"] == "complete" and len(complete["rounds"]) == 12  # Local: two worlds × two sides × three baselines.
    identities = {(row["baseline"], row["pair"], row["side"]) for row in complete["rounds"]}  # Local: all unique planned rounds appear once.
    assert len(identities) == 12 and complete["rounds"][0] == original  # Local: partial replay neither erases nor duplicates evidence.
    assert all(row["learner_plants"] == (1 if row["side"] == 0 else 0) for row in complete["rounds"])  # Local: learner attribution differs from round totals.
    count = TwoTickSimulation.constructions  # Local: capture simulation allocation count before idempotent resume.
    repeated = evaluation.evaluate_policy(ZeroPolicy(), output, pairs=2, num_envs=2, resume=True, identity="test-policy")
    assert repeated == complete and TwoTickSimulation.constructions == count  # Local: already complete evaluation performs no matches.
    with pytest.raises(ValueError, match="mismatch"):  # Local: altered model identity must reject existing raw rounds.
        evaluation.evaluate_policy(ZeroPolicy(), output, pairs=2, num_envs=2, resume=True, identity="changed-policy")
    with pytest.raises(ValueError, match="mismatch"):  # Local: altered seed protocol also rejects existing evidence.
        evaluation.evaluate_policy(ZeroPolicy(), output, pairs=2, num_envs=2, seed_base=3_000_000, resume=True, identity="test-policy")


def test_policy_identity_hashes_checkpoint_contents(tmp_path):
    """Local: reusing a filename cannot spoof policy identity. Global: resumed tournaments evaluate one immutable model."""
    checkpoint = tmp_path / "model.pt"  # Local: no torch loading is needed to validate byte-level identity.
    checkpoint.write_bytes(b"first checkpoint")  # Local: establish initial file content.
    policy = SimpleNamespace(path=str(checkpoint))  # Local: production inference exposes the same path attribute.
    original = evaluation.policy_identity(policy, None, True)  # Local: hash file bytes through the production helper.
    checkpoint.write_bytes(b"changed checkpoint")  # Local: simulate replacing weights at an unchanged filename.
    assert evaluation.policy_identity(policy, None, True) != original  # Local: content changes invalidate resume compatibility.
