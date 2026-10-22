"""Local: registered phase dispatch; global: independent workers share one immutable deadline.

Process API: https://docs.python.org/3.12/library/subprocess.html .
Telemetry API: https://psutil.io/ . Scheduling/selection rules are engineering choices
registered before optimization in docs/duel-protocol.md; no automatic reward tuning occurs here.
"""

import argparse  # Local: the hidden worker entry point receives only a published config/index.
from dataclasses import replace  # Local: resume changes location of saved state, never the learning recipe.
from datetime import datetime, timezone  # Local: retain process identity and elapsed wall time.
import logging  # Local: complete terminal/file errors and phase markers.
from pathlib import Path  # Local: project-local logs and checkpoints.
import subprocess  # Local: independent learners use separate Python processes/RNGs.
import sys  # Local: reuse the locked interpreter rather than a different Python installation.
import time  # Local: polling yields CPU; elapsed includes imports/checkpoint work.

import psutil  # Local: maintained portable host CPU/memory metrics, avoiding handwritten Windows APIs.

from ministrike.experiments.registration import MANIFEST, read_json, remaining_budget, training_preflight
from ministrike.learning.config import TrainConfig
from ministrike.learning.trainer import train_run
from ministrike.records import configure_logging, timestamp, write_json


def host_sample():
    """Local: record CPU/RAM/GPU evidence; global: throughput and headroom decide hardware, not utilization alone."""
    memory = psutil.virtual_memory()  # Local: available host RAM is meaningful across Windows caches.
    sample = {"timestamp_utc": timestamp(), "cpu_percent": psutil.cpu_percent(),
              "ram_available_fraction": memory.available / memory.total}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        utilization, used, total = map(float, result.stdout.strip().splitlines()[0].split(","))
        sample.update(gpu_percent=utilization, gpu_used_mib=used, gpu_total_mib=total,
                      gpu_available_fraction=1 - used / total)  # Local: retain actual VRAM headroom.
    except (OSError, subprocess.SubprocessError, ValueError):
        sample["gpu_telemetry_unavailable"] = True  # Local: missing telemetry is not interpreted as zero usage.
    return sample


def recover_suffix(directory):
    """Local: archive uncheckpointed metric rows; global: resume never duplicates canonical update IDs."""
    from ministrike.learning.checkpoint import load_checkpoint
    import json  # Local: preserve exact raw lines, parsing only their update IDs.

    directory = Path(directory)
    model = directory / "model.pt"
    if not model.exists():
        raise FileNotFoundError("No checkpoint to resume; preserve this failed run rather than replace it")
    saved = load_checkpoint(model)
    metrics = directory / "metrics.jsonl"
    if metrics.exists():
        lines = metrics.read_text(encoding="utf-8").splitlines(keepends=True)
        keep, suffix = [], []
        for line in lines:
            try:
                target = keep if json.loads(line)["updates"] <= saved["updates"] else suffix
            except (ValueError, KeyError):
                target = suffix  # Local: preserve incomplete final records for interruption auditing.
            target.append(line)
        if suffix:
            archive = directory / "recovery" / timestamp()
            archive.mkdir(parents=True)
            (archive / "uncheckpointed-metrics.jsonl").write_text("".join(suffix), encoding="utf-8")
            metrics.write_text("".join(keep), encoding="utf-8")
    return str(model)  # Local: original clock remains in the shared manifest, including discarded work/downtime.


def run_group(config_path, indices, name, allow_resume=False):
    """Local: launch and supervise one independent group; global: preserve every failure and telemetry sample."""
    record = read_json(config_path)
    configs = [TrainConfig(**record["runs"][index], run_config=str(config_path)) for index in indices]
    result_path = MANIFEST.parent / f"{name}.json"
    if result_path.exists() and read_json(result_path).get("status") in {"complete", "failed"}:
        previous = read_json(result_path)
        if not allow_resume or all(item.get("complete") for item in previous.get("results", [{}])):
            return previous  # Local: never retrain completed results or repeat failed calibration automatically.
        write_json(result_path.with_name(result_path.stem + "-archived-" + timestamp() + ".json"), previous)
    for config in configs:
        training_preflight(config)  # Local: validate all children before any optimizer work starts.
    remaining_budget(start=True)  # Local: parent creates the unique clock before concurrent workers read it.
    started = time.monotonic()
    workers, samples = [], []  # Local: explicit handles limit cleanup to children launched by this group.
    receipt = {"name": name, "status": "running", "config": str(config_path), "indices": indices,
               "started_utc": datetime.now(timezone.utc).isoformat(), "workers": [], "telemetry": samples}
    state = read_json(MANIFEST)
    state["active_group"] = str(result_path)
    state["status"] = "calibration" if configs[0].registration == "A-CAL" else configs[0].registration
    write_json(MANIFEST, state)  # Local: heartbeat can resolve current identities rather than historical PIDs.
    deadline = started + min(remaining_budget(), max(config.max_seconds for config in configs) + 15)
    try:
        for index, config in zip(indices, configs, strict=True):
            directory = Path(config.output_dir)
            command = [sys.executable, "-u", "-m", "ministrike.experiments.staged", "--config", str(config_path), "--index", str(index)]
            if allow_resume and (directory / "model.pt").exists():
                command.extend(["--resume", recover_suffix(directory)])  # Local: retain checkpoints and audit suffixes first.
            stream = (directory / f"worker-{timestamp()}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workers.append((process, stream, directory))
            receipt["workers"].append({"pid": process.pid, "command": command, "output_dir": str(directory)})
        write_json(result_path, receipt)  # Local: process identity is durably recorded before monitoring.
        while any(process.poll() is None for process, _, _ in workers):
            samples.append(host_sample())
            if time.monotonic() >= deadline:
                raise TimeoutError("Registered group or absolute deadline expired")  # Local: no deadline extension.
            write_json(result_path, receipt)
            time.sleep(2)  # Local: modest telemetry frequency, no busy polling.
        receipt["results"] = [read_json(directory / "result.json") if (directory / "result.json").exists()
                              else {"complete": False, "failure": "worker produced no result"} for _, _, directory in workers]
        receipt["exit_codes"] = [process.returncode for process, _, _ in workers]
        receipt["status"] = "complete" if all(code == 0 for code in receipt["exit_codes"]) else "failed"
    except Exception as error:
        receipt.update(status="failed", error=repr(error))  # Local: persist failure before propagating to the caller.
        logging.exception("Registered group failed: %s", name)
    finally:
        for process, stream, _ in workers:
            if process.poll() is None:
                process.kill()  # Local: bound only our expired worker; valid periodic checkpoints remain recoverable.
                process.wait()
            stream.close()
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["aggregate_arena_steps_per_second"] = sum(item.get("frames", 0) for item in receipt.get("results", [])) / receipt["wall_seconds"]
        write_json(result_path, receipt)
    return receipt


def execute_registered(config_path, index=0, suite=False, calibrate=False, resume=None):
    """Local: readable orchestration entry point; global: no implicit future budgets or experiment variants."""
    if config_path is None:
        raise ValueError("Provide --run-config with the published resolved configuration; read docs/experiment-roadmap.md")
    record = read_json(config_path)
    if calibrate:
        return [run_group(config_path, group["indices"], f"calibration/{group['name']}") for group in record["groups"]]
    if suite:
        if any(row["registration"] != "A-CONFIRM" for row in record["runs"]):
            raise ValueError("train --suite requires the three registered confirmation seeds")
        if sorted(row["seed"] for row in record["runs"]) != [5101, 5102, 5103]:
            raise ValueError("Confirmation seed identities differ from the approved registration")
        groups = [list(range(3))] if record["concurrency"] == 3 else [[0], [1], [2]]
        return [run_group(config_path, group, "confirmation/group-" + str(group[0]), bool(resume)) for group in groups]
    config = TrainConfig(**record["runs"][index], run_config=str(config_path))
    if resume:
        config = replace(config, resume=recover_suffix(Path(resume).parent))
    return train_run(config)  # Local: individual pilot or confirmation run still passes trainer preflight.


def main():
    """Local: fresh worker process; global: imports alone never start training or clocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = TrainConfig(**read_json(args.config)["runs"][args.index], run_config=str(args.config))
    configure_logging(Path(config.output_dir) / "logs")
    try:
        train_run(replace(config, resume=str(args.resume) if args.resume else None))
    except Exception:
        logging.exception("Registered training worker failed")
        raise


if __name__ == "__main__":
    main()  # Local: Windows worker entry point is import safe.
