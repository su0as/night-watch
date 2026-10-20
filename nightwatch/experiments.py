"""The set of recorded runs the evaluation uses. Each one is a separate process, a few run at
once. Clean runs matter as much as broken ones: they are where false wake-ups come from."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MATRIX: list[tuple[str, str, int]] = [
    ("healthy-s1", "none", 1),
    ("healthy-s2", "none", 2),
    ("healthy-s3", "none", 3),
    ("healthy-s4", "none", 4),
    ("flip-s1", "flip_bug", 1),
    ("flip-s2", "flip_bug", 2),
    ("spin-s1", "spin_bug", 1),
    ("lr-s1", "lr_blowup@200000", 1),
    ("lr-s2", "lr_blowup@150000", 2),
    ("physics-s1", "physics_blowup@200000", 1),
    ("stall-s1", "stall@250000:hold_s=900", 1),
    ("crash-s1", "crash@250000", 1),
]

SMOKE_MATRIX: list[tuple[str, str, int]] = [
    ("smoke-healthy", "none", 1),
    ("smoke-flip", "flip_bug", 1),
    ("smoke-lr", "lr_blowup@20000", 2),
    ("smoke-crash", "crash@30000", 1),
]


def physical_cores() -> int:
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or os.cpu_count() or 2)
    except Exception:  # noqa: BLE001
        return max(1, (os.cpu_count() or 2) // 2)


def threads_per_job(jobs: int, threads: int | None = None) -> int:
    """Split the physical cores between the parallel runs instead of letting each grab all."""
    if threads:
        return max(1, int(threads))
    return max(1, physical_cores() // max(1, jobs))


def run_matrix(out_dir: str, matrix: list[tuple[str, str, int]], steps: int, jobs: int, env_id: str,
               preset: str, frame_every: int, only: list[str] | None = None,
               threads: int | None = None) -> list[dict]:
    out = Path(out_dir)
    n_threads = threads_per_job(jobs, threads)
    out.mkdir(parents=True, exist_ok=True)
    todo = [m for m in matrix if not only or m[0] in only]
    todo = [m for m in todo if not (out / m[0] / "manifest.json").exists()]
    running: list[tuple[str, subprocess.Popen, float]] = []
    results: list[dict] = []
    print(f"[runs] {len(todo)} runs to do, {jobs} at a time, {steps:,} steps each, "
          f"{n_threads} torch thread(s) per run", flush=True)
    while todo or running:
        while todo and len(running) < jobs:
            name, fault, seed = todo.pop(0)
            log = open(out / f"{name}.log", "w", encoding="utf-8")
            cmd = [sys.executable, "-m", "nightwatch", "train", "--env", env_id, "--steps", str(steps),
                   "--seed", str(seed), "--fault", fault, "--name", name, "--runs-dir", str(out),
                   "--preset", preset, "--frame-every", str(frame_every), "--threads", str(n_threads)]
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, p, time.time()))
            print(f"[runs] started {name} ({fault}, seed {seed})", flush=True)
        time.sleep(5)
        for item in list(running):
            name, p, t0 = item
            code = p.poll()
            if code is not None:
                running.remove(item)
                results.append({"name": name, "exit": code, "minutes": round((time.time() - t0) / 60, 1)})
                print(f"[runs] {name} exited with {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return results
