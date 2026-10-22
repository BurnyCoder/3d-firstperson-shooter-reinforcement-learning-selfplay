"""Local: profile production MAPPO phases; global: hardware calibration, never quality evidence.

Run each candidate in a fresh process so CPU thread pools/CUDA allocation history do not leak.
The warmup and optional cProfile collection are excluded from throughput. Measured updates
include collection, real PPO, actor synchronization, ordinary snapshot cadence, and one final
checkpoint. Thus changing arena count also changes rollout size; this is a hardware probe,
not permission to change an ongoing experiment's frozen training protocol.

Sources: https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html
https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html
https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html
https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html
https://docs.python.org/3.12/library/profile.html
"""

import argparse  # Local: one candidate per CLI invocation; global: independent hardware trials.
import copy  # Local: separate CPU collection weights; global: match the production trainer.
import cProfile  # Local: optional function-call diagnostics; global: locate collection bottlenecks.
from dataclasses import asdict  # Local: persist all settings; global: reproducible comparisons.
import io  # Local: retain complete profiler text; global: the log and artifact share evidence.
import json  # Local: full structured phase records; global: no truncated metric values.
import logging  # Local: shared timestamped logging; global: inspectable terminal/file output.
from pathlib import Path  # Local: project-local artifacts; global: no external path discovery.
import pstats  # Local: sort the profiler's call table; global: identify cumulative CPU cost.
import random  # Local: fix the Python RNG; global: disjoint calibration seed 8002.
import time  # Local: monotonic high-resolution timings; global: measure completed work.
from types import SimpleNamespace  # Local: group phase dependencies; global: keep the wrapper readable.

import numpy as np  # Local: seed NumPy initialization; global: shared simulation reproducibility.
import torch  # Local: production optimizer/device synchronization; global: real MAPPO measurements.

from ministrike.learning.checkpoint import save_checkpoint  # Reuse the safe atomic serializer.
from ministrike.learning.config import TrainConfig  # Reuse identical architecture/hyperparameter settings.
from ministrike.learning.environment import ArenaEnv  # Measure authoritative batched arenas.
from ministrike.learning.models import ActorNetwork, CentralCritic, make_actor, make_critic
from ministrike.learning.objectives import make_objectives  # Reuse TorchRL loss and advantage machinery.
from ministrike.learning.opponents import SnapshotPool  # Reuse immutable per-round opponents.
from ministrike.learning.trainer import _capture, collect_batch, optimize_batch  # No substitute PPO implementation.
from ministrike.records import configure_logging, source_revision, timestamp, write_json

LOGGER = logging.getLogger(__name__)  # The CLI installs both timestamped output handlers.


def timed_phase(device, function, *args):
    """Local: time completed CPU/CUDA work; global: asynchronous kernels cannot inflate FPS."""
    if device.startswith("cuda"):  # CPU trials have no accelerator work to synchronize.
        torch.cuda.synchronize(device)  # Finish earlier kernels before starting this interval.
    started = time.perf_counter()  # A monotonic wall clock includes Python and native work.
    value = function(*args)  # Execute the unchanged production phase.
    if device.startswith("cuda"):  # Kernel enqueue time alone is not execution time.
        torch.cuda.synchronize(device)  # Include completion of this phase's CUDA work.
    return value, time.perf_counter() - started  # Return data and timing without recomputation.


def initialize_calibration(config, snapshots):
    """Local: construct the real learner once; global: isolate setup from steady-state throughput."""
    config.validate()  # Reuse production validation before allocating any arenas.
    if config.device.startswith("cuda") and not torch.cuda.is_available():  # Reject false GPU measurements.
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config.torch_threads)  # Set intra-op threads before tensor computation.
    random.seed(config.seed)  # Keep standard-library randomness independent from the main experiment.
    np.random.seed(config.seed)  # Fix any global NumPy draws used by dependencies.
    torch.manual_seed(config.seed)  # Fix network initialization and categorical samples.
    actor = ActorNetwork().to(config.device)  # Same local actor and selected optimization device.
    critic = CentralCritic().to(config.device)  # Same centralized five-player value network.
    rollout = copy.deepcopy(actor).cpu()  # Production collection performs policy inference on CPU.
    pool = SnapshotPool(rollout, config.seed + 2003)  # Same opponent stream separation as train_run.
    for identity in range(-snapshots, 0):  # Distinct identities approximate a mature pool's grouped inference overhead.
        pool.add(rollout, identity)  # Identical initial weights are calibration copies, not trained opponents.
    env = ArenaEnv(config.num_envs, config.seed, config.condition, pool, scenario="standard")
    env.reset()  # Start ordinary map rounds; do not benchmark only the easier close-combat curriculum.
    parameters = list(actor.parameters()) + list(critic.parameters())  # Optimize the real actor and critic.
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)  # Preserve the production optimizer.
    loss, estimator = make_objectives(make_actor(actor), make_critic(critic), config)  # Library PPO/GAE.
    return SimpleNamespace(actor=actor, critic=critic, rollout=rollout, rollout_actor=make_actor(rollout),
                           pool=pool, env=env, parameters=parameters, optimizer=optimizer,
                           loss=loss, estimator=estimator)  # Named fields keep orchestration comprehensible.


def measure_update(state, config, update, checkpoint, elapsed):
    """Local: time each production phase; global: report end-to-end rather than GPU-only throughput."""
    started = time.perf_counter()  # Include all orchestration and synchronization within this update.
    batch, collection = timed_phase(config.device, collect_batch, state.env, state.rollout_actor,
                                    config.rollout_steps)  # Preserve full temporal team trajectories.
    metrics, optimize = timed_phase(config.device, optimize_batch, batch, state.loss, state.estimator,
                                   state.optimizer, state.parameters, config)  # Real finite-gradient PPO updates.
    _, synchronize = timed_phase(config.device, state.rollout.load_state_dict, state.actor.state_dict())
    snapshot_started = time.perf_counter()  # Account for immutable snapshot copies separately from PPO.
    if update % config.snapshot_interval == 0:  # Match production opponent refresh cadence.
        state.pool.add(state.rollout, update)  # Freeze only a complete updated learner.
        state.pool.prune(state.env.opponent_ids)  # Preserve opponents still used by unfinished rounds.
    snapshot = time.perf_counter() - snapshot_started  # Usually negligible outside snapshot updates.
    checkpoint_seconds = 0.0  # Ordinary intermediate updates need not serialize large trajectories.
    if checkpoint or update % config.snapshot_interval == 0:  # Include the final checkpoint in each candidate.
        checkpoint_started = time.perf_counter()  # Capture includes CPU transfer and simulator/RNG copies.
        capture = _capture(config, state.actor, state.critic, state.optimizer, state.pool, state.env,
                           update * config.num_envs * config.rollout_steps, update, elapsed)
        save_checkpoint(Path(config.output_dir) / "calibration.pt", capture)  # Never a deployment checkpoint.
        if config.device.startswith("cuda"):  # Complete device-to-host work before stopping the timer.
            torch.cuda.synchronize(config.device)
        checkpoint_seconds = time.perf_counter() - checkpoint_started  # Include serialization and replacement.
    total = time.perf_counter() - started  # Includes every measured phase and wrapper overhead.
    return {"timestamp_utc": timestamp(), "update": update, "collection_seconds": collection,
            "optimization_seconds": optimize, "actor_sync_seconds": synchronize,
            "snapshot_seconds": snapshot, "checkpoint_seconds": checkpoint_seconds,
            "total_seconds": total, "arena_frames": config.num_envs * config.rollout_steps,
            "arena_frames_per_second": config.num_envs * config.rollout_steps / total,
            **metrics}  # Persist all finite loss diagnostics without treating them as quality results.


def profile_collection(state, config):
    """Local: profile one additional collection; global: profiler overhead never contaminates FPS."""
    profiler = cProfile.Profile()  # Standard deterministic profiler records Python and extension call sites.
    profiler.runcall(collect_batch, state.env, state.rollout_actor, config.rollout_steps)  # No extra optimizer update.
    profiler.dump_stats(str(Path(config.output_dir) / "collection.prof"))  # Save the full untruncated call graph.
    output = io.StringIO()  # Format one reusable text artifact and terminal log entry.
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(30)  # Highlight the top cumulative costs.
    text = output.getvalue()  # The binary profile above retains every call beyond this human-readable top thirty.
    (Path(config.output_dir) / "collection-profile.txt").write_text(text, encoding="utf-8")
    LOGGER.info("collection_profile_top30\n%s", text)  # Mirror diagnostic text without truncating the logger.


def benchmark(config, updates=2, collection_profile=False, snapshots=8):
    """Local: warm up, measure, then profile; global: disposable hardware calibration only."""
    if updates < 1:  # A throughput estimate requires at least one measured production update.
        raise ValueError("updates must be positive")
    if not 0 <= snapshots <= 8:  # Use at most the production historical capacity; negative IDs avoid update collisions.
        raise ValueError("snapshots must be between zero and eight additional initial-weight copies")
    directory = Path(config.output_dir)  # Keep each candidate's artifacts in an independent directory.
    directory.mkdir(parents=True, exist_ok=True)  # Create the caller-selected project-local destination.
    if (directory / "performance.json").exists() or (directory / "config.json").exists():
        raise FileExistsError(f"Calibration already exists: {directory}; use a fresh output directory")
    write_json(directory / "config.json", asdict(config))  # Persist full settings before any optimizer work.
    LOGGER.info("hardware_calibration_started %s", json.dumps({"config": asdict(config), "snapshot_copies": snapshots}))
    state = initialize_calibration(config, snapshots)  # Setup costs are excluded and must not masquerade as steady state.
    try:  # Always release TorchRL resources even if collection or optimization fails.
        warmup = measure_update(state, config, 1, False, 0.0)  # One real update initializes caches and optimizer state.
        LOGGER.info("hardware_calibration_warmup %s", json.dumps(warmup))  # Preserve excluded warmup evidence.
        if config.device.startswith("cuda"):  # Peak allocation starts after one-time CUDA initialization.
            torch.cuda.reset_peak_memory_stats(config.device)  # Reset peaks to current live allocation baseline.
        rows = []  # Measured rows exclude warmup and the optional instrumented collection.
        started = time.perf_counter()  # Whole loop includes per-update logging and metrics persistence.
        for index in range(updates):  # Candidate runs use the caller's bounded number of measured updates.
            row = measure_update(state, config, index + 2, index == updates - 1,
                                 warmup["total_seconds"] + time.perf_counter() - started)
            rows.append(row)  # Retain each individual sample instead of reporting only a favorable one.
            with (directory / "timings.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")  # Flush one complete measurement on close.
            LOGGER.info("hardware_calibration_update %s", json.dumps(row))  # Real-time timestamped file/terminal record.
        elapsed = time.perf_counter() - started  # End-to-end cost includes final checkpoint and logging.
        result = {"purpose": "throwaway hardware calibration; no policy-quality evaluation",
                  "timestamp_utc": timestamp(), "config": asdict(config), "scenario": "standard",
                  "warmup": warmup, "timings": rows, "measured_updates": updates,
                  "measured_arena_frames": updates * config.num_envs * config.rollout_steps,
                  "elapsed_seconds": elapsed,
                  "arena_frames_per_second": updates * config.num_envs * config.rollout_steps / elapsed,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(config.device)
                      if config.device.startswith("cuda") else 0,
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(config.device)
                      if config.device.startswith("cuda") else 0,
                  "device_name": torch.cuda.get_device_name(config.device)
                      if config.device.startswith("cuda") else "CPU",
                  "torch_version": torch.__version__, "collection_profile_requested": collection_profile}
        result["snapshot_copies"] = snapshots  # These immutable identities share initial weights, not independent training.
        write_json(directory / "performance.json", result)  # The filename cannot be confused with experiment result.json.
        LOGGER.info("hardware_calibration_finished %s", json.dumps(result))  # Keep complete aggregate provenance.
        if collection_profile:  # Diagnostic profiler overhead belongs outside the throughput interval.
            profile_collection(state, config)
        return result  # Orchestrators can compare completed candidates without parsing console text.
    finally:
        state.env.close()  # Follow EnvBase's resource lifecycle on both successful and failed measurements.


def main():
    """Local: expose one hardware candidate; global: leave the active experiment untouched."""
    parser = argparse.ArgumentParser(description=__doc__)  # CLI help states the scope and scientific caveats.
    parser.add_argument("--num-envs", type=int, default=16)  # Typical sweep candidates: 16, 32, 64, 128 arenas.
    parser.add_argument("--torch-threads", type=int, default=4)  # Typical CPU intra-op candidates: 1, 2, 4.
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")  # Never silently substitute CPU for GPU.
    parser.add_argument("--updates", type=int, default=2)  # Two measured updates follow one excluded warmup.
    parser.add_argument("--seed", type=int, default=8002)  # Fresh calibration seed; concurrent workers can use 8003.
    parser.add_argument("--snapshots", type=int, default=8)  # Additional initial-weight copies stress grouped opponent inference.
    parser.add_argument("--condition", choices=("selfplay", "fixed"), default="selfplay")  # Measure both opponent costs.
    parser.add_argument("--rollout-steps", type=int, default=128)  # Production trajectory length unless explicitly overridden.
    parser.add_argument("--output", type=Path, default=Path("runs/performance") / timestamp())  # No collisions by default.
    parser.add_argument("--profile-collection", action="store_true")  # Optional full cProfile collection diagnostic.
    args = parser.parse_args()  # Parse before allocating models or touching CUDA.
    configure_logging(args.output / "logs")  # Timestamped terminal/file logging also captures dependency warnings.
    config = TrainConfig(output_dir=str(args.output), condition=args.condition, seed=args.seed,
                         num_envs=args.num_envs, total_frames=(args.updates + 1) * args.num_envs * args.rollout_steps,
                         device=args.device, rollout_steps=args.rollout_steps, torch_threads=args.torch_threads,
                         source_revision=source_revision())  # Preserve production defaults for all other hyperparameters.
    try:  # Retain complete tracebacks for any failed hardware candidate.
        benchmark(config, args.updates, args.profile_collection, args.snapshots)  # Delegate readable phases to reusable functions.
    except Exception:  # Re-raising preserves a nonzero process exit code for sweep orchestration.
        LOGGER.exception("hardware_calibration_failed")  # No calibration failures are silently discarded.
        raise


if __name__ == "__main__":  # Importing this module never starts a benchmark or changes the active suite.
    main()  # Explicit module invocation is the only CLI execution path.
