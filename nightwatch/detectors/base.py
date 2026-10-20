"""What a watcher sees, and what it says back."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Protocol

from ..runlog import RunReader


@dataclass
class Signal:
    """One finding from one detector at one moment.

    level: "wake" (worth a phone buzz), "note" (put it in the morning report), "ok".
    """

    source: str
    kind: str
    level: str
    score: float
    message: str
    step: int = -1
    t: float = 0.0
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass
class Observation:
    run_dir: str
    run_id: str
    now: float
    live: bool
    manifest: dict
    metrics: list[dict]
    frames: list[dict]
    heartbeat: dict | None
    process_alive: bool | None

    @property
    def status(self) -> str:
        return str(self.manifest.get("status", "running"))

    @property
    def env_id(self) -> str:
        return str(self.manifest.get("env_id", ""))

    @property
    def last_metric_t(self) -> float | None:
        return float(self.metrics[-1]["t"]) if self.metrics else None

    @property
    def last_step(self) -> int:
        if self.metrics:
            return int(self.metrics[-1].get("step", 0))
        if self.frames:
            return int(self.frames[-1].get("step", 0))
        return 0

    @property
    def expected_interval_s(self) -> float | None:
        ts = [float(r["t"]) for r in self.metrics[-21:]]
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        return median(gaps) if gaps else None


class Detector(Protocol):
    name: str

    def check(self, obs: Observation) -> list[Signal]:
        ...


def _pid_alive(pid: int) -> bool | None:
    """True/False if we can tell, None if we cannot."""
    pid = int(pid)
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001  psutil missing or broken: fall back to the OS
        pass
    if os.name == "nt":
        # Never os.kill(pid, 0) on Windows: it calls TerminateProcess.
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(handle, ctypes.byref(code))
            k32.CloseHandle(handle)
            return code.value == 259  # STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def observe(run_dir: str, now: float | None = None, live: bool = True) -> Observation:
    """Snapshot of a run as of `now`.

    live=True  reads everything and checks whether the trainer process is still alive.
    live=False is the evaluation replay: only data with t <= now is visible, and the
               run's final status only becomes visible once its ended_at has passed.
    """
    reader = RunReader(run_dir)
    now = time.time() if now is None else float(now)
    manifest = reader.manifest()
    if live:
        metrics = reader.metrics()
        frames = reader.frames()
        hb = reader.heartbeat()
        alive = None
        same_host = manifest.get("host") in (None, platform.node())
        if hb and hb.get("pid") and same_host and manifest.get("status") == "running":
            alive = _pid_alive(hb["pid"])
    else:
        metrics = reader.metrics(until_t=now)
        frames = reader.frames(until_t=now)
        hb = None
        alive = None
        ended = manifest.get("ended_at")
        if ended is None or float(ended) > now:
            manifest = {**manifest, "status": "running"}
            manifest.pop("ended_at", None)
    return Observation(run_dir=str(run_dir), run_id=reader.run_id, now=now, live=live,
                       manifest=manifest, metrics=metrics, frames=frames, heartbeat=hb,
                       process_alive=alive)
