"""The baseline: the alert rules people usually set up in their experiment tracker.

Reward dropped 30% from its best, KL divergence spiked, value loss exploded, policy
noise blew up. Sensible, cheap, and blind to a robot that is getting *more* reward by
doing the wrong thing.
"""

from __future__ import annotations

import math
from statistics import median

from ..features import KEYS
from .base import Observation, Signal


def _num(row: dict, key: str) -> float | None:
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


class Thresholds:
    name = "thresholds"

    def __init__(self, drop_frac: float = 0.3, kl_max: float = 0.15, vloss_factor: float = 20.0,
                 std_max: float = 5.0, ev_min: float = -1.0, warmup_rows: int = 10):
        self.drop_frac = drop_frac
        self.kl_max = kl_max
        self.vloss_factor = vloss_factor
        self.std_max = std_max
        self.ev_min = ev_min
        self.warmup_rows = warmup_rows

    def check(self, obs: Observation) -> list[Signal]:
        rows = obs.metrics
        if len(rows) <= self.warmup_rows:
            return []
        last = rows[-1]
        step = int(last.get("step", 0))
        out: list[Signal] = []

        rews = [v for v in (_num(r, KEYS["rew"]) for r in rows) if v is not None]
        if len(rews) > self.warmup_rows:
            # "Lost 30% of the progress made so far". A plain "30% below best" misfires while
            # rewards are still negative early in training, and nobody would keep that alert.
            start = sum(rews[:3]) / 3
            best, now_r = max(rews), rews[-1]
            gained = best - start
            meaningful = gained > 0.1 * max(abs(best), abs(start), 1.0)
            if meaningful and best - now_r > self.drop_frac * gained:
                out.append(Signal("thresholds", "reward_drop", "wake", (best - now_r) / gained,
                                  f"Reward fell from {best:.0f} to {now_r:.0f} "
                                  f"(lost {100 * (best - now_r) / gained:.0f}% of the gains).", step, obs.now))

        kl = _num(last, KEYS["kl"])
        if kl is not None and kl > self.kl_max:
            out.append(Signal("thresholds", "kl_spike", "wake", kl,
                              f"Approx KL {kl:.3f} is above {self.kl_max}.", step, obs.now))

        vls = [v for v in (_num(r, KEYS["vloss"]) for r in rows[:-1]) if v is not None]
        vl = _num(last, KEYS["vloss"])
        if vl is not None and len(vls) >= self.warmup_rows:
            base = median(vls)
            if base > 0 and vl > self.vloss_factor * base:
                out.append(Signal("thresholds", "value_loss_spike", "wake", vl / base,
                                  f"Value loss {vl:.3g} is {vl / base:.0f}x its usual level.", step, obs.now))

        std = _num(last, KEYS["std"])
        if std is not None and std > self.std_max:
            out.append(Signal("thresholds", "std_explosion", "wake", std,
                              f"Policy noise (std) is {std:.2f}.", step, obs.now))

        ev = _num(last, KEYS["ev"])
        if ev is not None and ev < self.ev_min:
            out.append(Signal("thresholds", "ev_collapse", "wake", -ev,
                              f"Explained variance dropped to {ev:.2f}.", step, obs.now))
        return out
