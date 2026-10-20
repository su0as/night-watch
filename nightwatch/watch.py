"""The plain watcher loop: no Temporal, just a while loop and time.sleep.

It works, and it is the baseline for the durability test: kill this process at 3 a.m.
and the watch is simply over. The Temporal version (temporal_app/) picks up where it
left off.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path

from . import tracing
from .config import Settings
from .detectors.base import observe
from .notify import Notifier
from .policy import PolicyConfig, PolicyState, decide
from .report import write_morning_report
from .runtime import WatchContext


def parse_until(spec: str, now: float | None = None) -> float:
    """'07:30' (next occurrence, local time), '+8h', '+45m', or an ISO timestamp."""
    now = time.time() if now is None else now
    spec = spec.strip()
    m = re.fullmatch(r"\+(\d+(?:\.\d+)?)([hm])", spec)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        return now + n * (3600 if unit == "h" else 60)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", spec)
    if m:
        base = dt.datetime.fromtimestamp(now)
        target = base.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if target.timestamp() <= now:
            target += dt.timedelta(days=1)
        return target.timestamp()
    return dt.datetime.fromisoformat(spec).timestamp()


def watch_plain(run_dir: str, profile: str, until_ts: float, settings: Settings,
                every_s: float | None = None) -> dict:
    tracing.init(settings.sentry_dsn)
    ctx = WatchContext(settings, profile)
    notifier = Notifier(settings)
    cfg = PolicyConfig(max_wakes=settings.max_wakes, snooze_s=settings.snooze_min * 60)
    every = every_s or settings.check_every_s
    state = PolicyState()
    started = time.time()
    run_dir = str(Path(run_dir).resolve())
    notifier.send("Night Watch is on", f"Watching {Path(run_dir).name} until "
                  f"{time.strftime('%H:%M', time.localtime(until_ts))} ({profile}). Sleep well.")
    while time.time() < until_ts:
        with tracing.transaction(f"check {Path(run_dir).name}"):
            signals, status, step = ctx.check(run_dir)
        alerts, state = decide(state, signals, time.time(), cfg)
        for a in alerts:
            image = str(Path(run_dir) / a.evidence["sheet"]) if a.evidence.get("sheet") else None
            notifier.send(f"Night Watch: {a.kind.replace('_', ' ')}", a.message, urgent=True, image=image)
        print(f"[watch] step {step:,} status={status} signals={len(signals)} wakes={len(state.wakes)} "
              f"notes={len(state.notes)}", flush=True)
        if status in ("finished", "error"):
            break
        obs = observe(run_dir, live=True)
        if obs.process_alive is False and any(a.kind == "dead" for a in state.wakes):
            break
        time.sleep(max(1.0, min(every, until_ts - time.time())))
    rep = write_morning_report(run_dir, state, client=ctx.client, text_model=settings.text_model,
                               watch_started=started, watch_ended=time.time(), profile=profile)
    notifier.send("Night Watch: morning report", rep["summary"], urgent=False, image=rep.get("gif"))
    print(f"[watch] report: {rep['markdown']}", flush=True)
    return rep
