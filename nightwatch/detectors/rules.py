"""Things that do not need a model: NaNs, a stalled log, a dead process, a crash.

Every watcher profile includes these, so comparisons between profiles are about what
the models add on top.
"""

from __future__ import annotations

import math

from .base import Observation, Signal


class Rules:
    name = "rules"

    def __init__(self, stall_factor: float = 6.0, min_stall_s: float = 600.0, startup_grace_s: float = 1200.0):
        self.stall_factor = stall_factor
        self.min_stall_s = min_stall_s
        self.startup_grace_s = startup_grace_s

    def check(self, obs: Observation) -> list[Signal]:
        out: list[Signal] = []
        status = obs.status
        step = obs.last_step
        if status == "error":
            err = obs.manifest.get("error", "the trainer reported an error")
            out.append(Signal("rules", "crashed", "wake", 1.0, f"Training crashed: {err}", step, obs.now))
            return out
        if status == "finished":
            out.append(Signal("rules", "finished", "note", 0.0, f"Training finished at step {step:,}.", step, obs.now))
            return out

        if obs.metrics:
            last = obs.metrics[-1]
            bad = [k for k, v in last.items() if isinstance(v, float) and not math.isfinite(v)]
            if bad:
                out.append(Signal("rules", "nan", "wake", 1.0,
                                  f"Non-finite values in the training log: {', '.join(sorted(bad)[:4])}.",
                                  step, obs.now))

        started = obs.manifest.get("started_at")
        last_t = obs.last_metric_t
        if last_t is None:
            if started and obs.now - float(started) > self.startup_grace_s:
                out.append(Signal("rules", "no_progress", "wake", 1.0,
                                  "No training metrics since the run started.", step, obs.now))
            return out

        interval = obs.expected_interval_s or 60.0
        limit = max(self.stall_factor * interval, self.min_stall_s)
        silent = obs.now - last_t
        if silent > limit:
            minutes = silent / 60.0
            if obs.live and obs.process_alive is False:
                out.append(Signal("rules", "dead", "wake", 1.0,
                                  f"The trainer process is gone and has logged nothing for {minutes:.0f} min.",
                                  step, obs.now))
            else:
                out.append(Signal("rules", "stalled", "wake", min(1.0, silent / (2 * limit)),
                                  f"No new training metrics for {minutes:.0f} min (usually every {interval:.0f} s).",
                                  step, obs.now))
        return out
