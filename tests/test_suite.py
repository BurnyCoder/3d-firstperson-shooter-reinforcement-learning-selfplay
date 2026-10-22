"""Local: frozen experiment bookkeeping regressions. Global: incomplete compute must never become completed evidence."""

from concurrent.futures import Future  # Local: completed task handles exercise ordering without starting training processes.
from datetime import UTC, datetime, timedelta  # Local: give fixtures a valid shared deadline without running training.
import json  # Local: inspect durable parent-written manifests instead of relying only on returned state.
from types import SimpleNamespace  # Local: minimal transition fixtures isolate orchestration from physics.

import numpy as np  # Local: preserve real batched shapes in fast evaluation fixtures.
import pytest  # Local: assert mismatched evidence cannot be resumed.

from ministrike.experiments import evaluation, suite  # Local: monkeypatch only expensive execution boundaries.
from ministrike.experiments.suite import RUN_ORDER, run_suite  # Local: exercise the actual resume/completion wrapper.
from ministrike.records import write_json  # Local: use the production atomic evidence format.
from ministrike.sim.constants import OBS_DIM  # Local: fixture policies receive the actual observation contract.


@pytest.mark.parametrize("workers", [1, 2, 4, 6])
def test_suite_schedules_each_identity_once_and_preserves_frozen_settings(tmp_path, monkeypatch, workers):
    """Local: exercise serial/concurrent scheduling without optimization. Global: parallel execution cannot change the experiment."""
    manifest = {"output_dir": str(tmp_path), "deadline_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "training_deadline_utc": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
                "num_envs": 16, "evaluation_pairs": 1, "total_frames": 2048, "per_run_seconds": 500,
                "device": "cuda", "runs": {}, "test_seed_base": 2_000_000, "source_revision": "frozen-source",
                "execution_workers": workers, "torch_threads": 1, "status": "execution_reconfigured"}  # Local: change execution topology only, keeping algorithm settings frozen.
    manifest_path = tmp_path / "suite.json"  # Local: all parent updates must target this one manifest.
    write_json(manifest_path, manifest)  # Local: resume avoids expensive device calibration.
    timed_out = {"condition": "selfplay", "seed": 111, "complete": False, "frames": 1024}  # Local: existing timeout is evidence, not permission to retrain.
    write_json(tmp_path / "selfplay_seed111" / "result.json", timed_out)  # Local: idempotency includes unsuccessful completed runs.
    for condition, seed in RUN_ORDER:  # Local: existing evaluations isolate scheduling from policy loading.
        for filename in ("evaluation.json", "initial_evaluation.json"):  # Local: both evaluation endpoints are already complete.
            write_json(tmp_path / f"{condition}_seed{seed}" / filename, {"status": "complete"})
    trained, submitted, logged = [], [], []  # Local: record execution, task identities, and worker log destinations.

    def fake_train(config, deadline_monotonic):
        """Local: return one synthetic completed run. Global: assert immutable training settings at the actual trainer boundary."""
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "training"  # Local: monitoring sees active work before any resumed optimizer task starts.
        trained.append((config.condition, config.seed))  # Local: duplicate dispatch would appear twice.
        assert (config.num_envs, config.total_frames, config.rollout_steps, config.minibatch_size, config.epochs) == (16, 2048, 128, 256, 3)
        assert (config.device, config.torch_threads, config.source_revision, config.max_seconds) == ("cuda", 1, "frozen-source", 500)
        assert 500 < deadline_monotonic - suite.time.monotonic() < 600  # Local: the training deadline is stricter than the overall hour.
        return {"condition": config.condition, "seed": config.seed, "complete": True, "frames": config.total_frames}

    class ImmediateExecutor:
        """Local: emulate independent task completion. Global: test parent bookkeeping without worker startup overhead."""

        def __init__(self, max_workers, mp_context):
            """Local: check explicit Windows-compatible process settings. Global: concurrency is bounded."""
            assert max_workers == workers and mp_context.get_start_method() == "spawn"  # Local: no implicit platform-dependent fork.

        def __enter__(self):
            """Local: match the standard executor context interface. Global: tests follow production scheduling."""
            return self  # Local: submit uses this fixture executor.

        def __exit__(self, *arguments):
            """Local: complete the context without suppressing errors. Global: failures remain visible to the suite."""
            return False  # Local: production failures must propagate.

        def submit(self, function, frozen, condition, seed):
            """Local: execute one task immediately. Global: future identity still separates work from parent writes."""
            submitted.append((condition, seed))  # Local: all six experiment identities must be submitted exactly once.
            future = Future()  # Local: real completed futures work with production result retrieval.
            future.set_result(function(frozen, condition, seed))  # Local: invoke the actual shared single-run helper.
            return future  # Local: the parent owns result persistence.

    monkeypatch.setattr(suite, "train_run", fake_train)  # Local: no optimizer or simulator work runs in this test.
    monkeypatch.setattr(suite, "configure_logging", lambda directory: logged.append(directory))  # Local: avoid replacing pytest's root handlers.
    monkeypatch.setattr(suite, "ProcessPoolExecutor", ImmediateExecutor)  # Local: no operating-system process is started.
    monkeypatch.setattr(suite, "as_completed", lambda futures: reversed(list(futures)))  # Local: out-of-order completion must retain correct names.
    result = run_suite(tmp_path, resume=manifest_path)  # Local: exercise the complete parent scheduling/completion phase.
    assert sorted(trained) == sorted(RUN_ORDER[1:])  # Local: existing timeout is reused and every other run executes once.
    assert len(result["runs"]) == 6 and result["runs"]["selfplay_seed111"] == timed_out  # Local: all scheduled outcomes remain visible.
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result  # Local: durable evidence exactly matches the returned result.
    assert result["status"] == "budget_exhausted"  # Local: parallelism cannot turn unequal-frame evidence into a complete experiment.
    assert submitted == (RUN_ORDER if workers > 1 else [])  # Local: serial mode must not use a process pool.
    assert len(logged) == (5 if workers > 1 else 0)  # Local: only newly executing child runs need separate log files.


def test_train_one_respects_exhausted_training_deadline(tmp_path, monkeypatch):
    """Local: queued work can expire before a worker starts. Global: the evaluation reserve cannot be spent on training."""
    manifest = {"output_dir": str(tmp_path), "deadline_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "training_deadline_utc": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()}  # Local: overall time remains, but training time is gone.
    monkeypatch.setattr(suite, "train_run", lambda *args, **kwargs: pytest.fail("expired training must not execute"))  # Local: no fictitious timeout checkpoint is produced.
    assert suite.train_one(manifest, "selfplay", 111) is None  # Local: absent evidence stays absent rather than a fabricated result.


def test_parallel_failure_is_recorded_before_worker_cleanup(tmp_path, monkeypatch):
    """Local: a failed future cancels work not yet started. Global: an observer sees failure while running siblings shut down."""
    manifest = {"output_dir": str(tmp_path), "deadline_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "num_envs": 16, "evaluation_pairs": 1, "total_frames": 2048, "per_run_seconds": 500,
                "device": "cuda", "runs": {}, "test_seed_base": 2_000_000, "source_revision": "frozen-source",
                "execution_workers": 2, "torch_threads": 1}  # Local: bypass calibration with a valid frozen protocol.
    manifest_path = tmp_path / "suite.json"  # Local: monitoring reads this file while executor cleanup may still be waiting.
    write_json(manifest_path, manifest)  # Local: establish the durable resume identity.
    pending = []  # Local: unfinished futures must be cancelled after the first failed result.

    class FailingExecutor:
        """Local: emulate one failed worker and five queued runs. Global: failure handling cannot rely on actual process timing."""

        def __init__(self, **kwargs):
            """Local: accept standard executor arguments. Global: isolate only result and cleanup behavior."""
            self.calls = 0  # Local: the first submission deterministically fails.

        def __enter__(self):
            """Local: supply the executor to the production context. Global: lifecycle ordering remains testable."""
            return self  # Local: submit below records synthetic worker futures.

        def __exit__(self, *arguments):
            """Local: inspect evidence before cleanup returns. Global: failed status is available without waiting for other runs."""
            assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"  # Local: write occurs before context exit can block.
            assert len(pending) == 5 and all(future.cancelled() for future in pending)  # Local: queued experiments must not launch after failure.
            return False  # Local: retain the original worker error for callers and logs.

        def submit(self, *arguments):
            """Local: create one exception and otherwise pending futures. Global: no simulator or optimizer executes."""
            future = Future()  # Local: cancellation follows the standard-library future behavior.
            if self.calls == 0:  # Local: fail the first task after successful submission.
                future.set_exception(RuntimeError("worker failed"))  # Local: result retrieval, rather than submission, raises.
            else:
                pending.append(future)  # Local: remaining tasks have not started and can be cancelled.
            self.calls += 1  # Local: every pre-specified identity is submitted once.
            return future  # Local: the parent handles asynchronous results.

    monkeypatch.setattr(suite, "ProcessPoolExecutor", FailingExecutor)  # Local: replace only process execution.
    monkeypatch.setattr(suite, "as_completed", lambda futures: iter(futures))  # Local: deliver the failed future first without waiting on pending work.
    with pytest.raises(RuntimeError, match="worker failed"):  # Local: failures remain visible to the CLI and unattended monitor.
        run_suite(tmp_path, resume=manifest_path)  # Local: exercise the real parent-side exception path.
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))  # Local: final durable status must still describe the failure.
    assert saved["status"] == "failed" and saved["runs"] == {}  # Local: failed or cancelled experiments do not become completed evidence.


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
