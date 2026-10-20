"""Replay a recorded run through a watcher, as if it were watching live.

At each check time the watcher only sees log rows and test episodes with t <= now, and the
run's final status only once its end time has passed. After the last log line of a run that
never finished (stall, crash) we keep checking for `tail_s` seconds, the way a real watcher
would keep looking at a silent run.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from ..detectors import run_detectors
from ..detectors.base import Observation
from ..policy import Alert, PolicyConfig, PolicyState, decide
from ..runlog import LabelReader, RunReader


@dataclass
class Preloaded:
    run_dir: str
    run_id: str
    manifest: dict
    ended_at: float | None
    metrics: list[dict]
    frames: list[dict]
    metric_ts: list[float]
    frame_ts: list[float]

    @classmethod
    def load(cls, run_dir: str | Path) -> "Preloaded":
        r = RunReader(run_dir)
        m = r.manifest()
        metrics = sorted(r.metrics(), key=lambda x: x["t"])
        frames = sorted(r.frames(), key=lambda x: x["t"])
        ended = m.get("ended_at") if m.get("status") in ("finished", "error") else None
        return cls(str(run_dir), r.run_id, m, float(ended) if ended else None, metrics, frames,
                   [float(x["t"]) for x in metrics], [float(x["t"]) for x in frames])

    def observe(self, now: float) -> Observation:
        man = dict(self.manifest)
        if self.ended_at is None or self.ended_at > now:
            man["status"] = "running"
            man.pop("ended_at", None)
        return Observation(run_dir=self.run_dir, run_id=self.run_id, now=now, live=False, manifest=man,
                           metrics=self.metrics[:bisect_right(self.metric_ts, now)],
                           frames=self.frames[:bisect_right(self.frame_ts, now)],
                           heartbeat=None, process_alive=None)


def check_times(pre: Preloaded, every_s: float = 120.0, tail_s: float = 3600.0) -> list[float]:
    times = sorted(set(pre.metric_ts) | set(pre.frame_ts))
    if not times:
        return []
    if pre.ended_at is not None:
        times.append(pre.ended_at + 1.0)
    else:
        last = times[-1]
        k = 1
        while k * every_s <= tail_s:
            times.append(last + k * every_s)
            k += 1
    return times


@dataclass
class ReplayResult:
    run_id: str
    profile: str
    wakes: list[Alert]
    notes: list[Alert]
    checks: int


def replay_run(run_dir: str | Path, profile: str, detectors: list, cfg: PolicyConfig | None = None,
               every_s: float = 120.0, tail_s: float = 3600.0) -> ReplayResult:
    pre = Preloaded.load(run_dir)
    state = PolicyState()
    for now in check_times(pre, every_s, tail_s):
        obs = pre.observe(now)
        _, state = decide(state, run_detectors(obs, detectors), now, cfg)
        if obs.status in ("finished", "error") and now >= (pre.ended_at or 0):
            break
    return ReplayResult(pre.run_id, profile, state.wakes, state.notes, state.checks)


def truth_by_step(run_dir: str | Path) -> dict[int, dict]:
    return {int(r["step"]): r for r in LabelReader(run_dir).truth()}
