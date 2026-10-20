"""On-disk format of a training run.

A run directory looks like this:

    runs/<name>/
        manifest.json        what is being trained (env, algo, seed, status). No labels.
        metrics.jsonl        one row per training log dump (SB3 keys + wall time "t")
        frames.jsonl         one row per test episode: step, t, contact sheet, eval return
        frames/              PNG snapshots of the robot
        heartbeat.json       pid, step, t; rewritten every few seconds by the trainer
        labels/              ground truth for evaluation ONLY. Watchers never read it.
            fault.json       which fault was injected and when
            truth.jsonl      simulator state during each test episode (pose, spin, NaNs)
        report/              morning report written by the watcher
        vision_cache.jsonl   Gemma's verdicts, so re-running the evaluation is free

The split matters: everything a watcher can see is in the top level, and everything that
would be cheating (the fault we injected, the true body orientation) lives in labels/.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def of(cls, root: str | os.PathLike) -> "RunPaths":
        return cls(Path(root))

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def frames_log(self) -> Path:
        return self.root / "frames.jsonl"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def heartbeat(self) -> Path:
        return self.root / "heartbeat.json"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def fault(self) -> Path:
        return self.labels_dir / "fault.json"

    @property
    def truth(self) -> Path:
        return self.labels_dir / "truth.jsonl"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    @property
    def vision_cache(self) -> Path:
        return self.root / "vision_cache.jsonl"


def _json_default(o: Any) -> Any:
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=_json_default)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file that another process may be appending to.

    A half-written last line is skipped instead of crashing the watcher.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_json_atomic(path: Path, obj: dict, retries: int = 20) -> None:
    """Write JSON via a temp file + os.replace.

    On Windows os.replace fails if a reader has the file open at that instant,
    so retry briefly instead of crashing the trainer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05 * (attempt + 1))
    return default


def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class RunReader:
    """Read-only view of what a watcher is allowed to see. Never touches labels/."""

    # Everything a real training script would log about itself. Nothing about injected faults.
    VISIBLE_MANIFEST_KEYS = ("run_id", "env_id", "algo", "seed", "started_at", "status", "ended_at", "preset",
                             "total_steps", "host", "error")

    def __init__(self, root: str | os.PathLike):
        self.paths = RunPaths.of(root)

    @property
    def run_id(self) -> str:
        return self.paths.name

    def manifest(self) -> dict:
        m = read_json(self.paths.manifest, default={}) or {}
        return {k: m[k] for k in self.VISIBLE_MANIFEST_KEYS if k in m}

    def metrics(self, until_t: float | None = None) -> list[dict]:
        rows = read_jsonl(self.paths.metrics)
        if until_t is not None:
            rows = [r for r in rows if r.get("t", 0) <= until_t]
        return rows

    def frames(self, until_t: float | None = None) -> list[dict]:
        rows = read_jsonl(self.paths.frames_log)
        if until_t is not None:
            rows = [r for r in rows if r.get("t", 0) <= until_t]
        return rows

    def heartbeat(self) -> dict | None:
        return read_json(self.paths.heartbeat, default=None)

    def resolve(self, rel: str) -> Path:
        return self.paths.root / rel


class LabelReader:
    """Ground truth for evaluation. Only the evaluation package should import this."""

    def __init__(self, root: str | os.PathLike):
        self.paths = RunPaths.of(root)

    def fault(self) -> dict:
        return read_json(self.paths.fault, default={"kind": "none", "at_step": 0}) or {"kind": "none", "at_step": 0}

    def truth(self) -> list[dict]:
        return read_jsonl(self.paths.truth)

    def full_manifest(self) -> dict:
        return read_json(self.paths.manifest, default={}) or {}


def list_runs(runs_dir: str | os.PathLike) -> list[Path]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "manifest.json").is_file())


def iter_numeric(row: dict, keys: Iterable[str]) -> list[float]:
    out = []
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out
