"""Turn the last few training-log rows into one row of numbers for TabPFN.

Pure Python on purpose: the same code runs inside the trainer's machine, a Temporal
activity and the offline evaluation, and it is easy to test without any ML stack.
Missing values are NaN; TabPFN handles missing values natively.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Sequence

KEYS = {
    "rew": "rollout/ep_rew_mean",
    "eplen": "rollout/ep_len_mean",
    "kl": "train/approx_kl",
    "clip": "train/clip_fraction",
    "ent": "train/entropy_loss",
    "vloss": "train/value_loss",
    "ev": "train/explained_variance",
    "std": "train/std",
    "fps": "time/fps",
    "lr": "train/learning_rate",
}

FEATURE_NAMES = [
    "rew_last", "rew_change_k", "rew_drop_from_max", "rew_slope_k", "rew_noise_k", "rew_gain_vs_early",
    "eplen_ratio",
    "kl_last", "kl_max_k", "kl_ratio",
    "clip_last",
    "ent_last", "ent_change_k",
    "vloss_log", "vloss_ratio",
    "ev_last",
    "std_last", "std_ratio",
    "fps_ratio", "lr_ratio",
    "nonfinite_k", "progress_log10",
]

NAN = float("nan")
DEFAULT_K = 8


def _val(row: dict, key: str) -> float:
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return NAN
    return float(v)


def _finite(xs: Sequence[float]) -> list[float]:
    return [x for x in xs if math.isfinite(x)]


def _safe_div(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return NAN
    return a / b


def _slope(ys: Sequence[float]) -> float:
    pts = [(i, y) for i, y in enumerate(ys) if math.isfinite(y)]
    if len(pts) < 2:
        return NAN
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return NAN
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def _std(xs: Sequence[float]) -> float:
    f = _finite(xs)
    if len(f) < 2:
        return NAN
    m = sum(f) / len(f)
    return math.sqrt(sum((x - m) ** 2 for x in f) / (len(f) - 1))


def _med(xs: Sequence[float]) -> float:
    f = _finite(xs)
    return median(f) if f else NAN


def series(rows: Sequence[dict], name: str) -> list[float]:
    return [_val(r, KEYS[name]) for r in rows]


def window_features(rows: Sequence[dict], k: int = DEFAULT_K) -> dict[str, float]:
    """Features describing the latest window of `k` rows, in context of the whole run so far."""
    if not rows:
        return {name: NAN for name in FEATURE_NAMES}
    win = rows[-k:]
    out: dict[str, float] = {}

    rew_all = series(rows, "rew")
    rew = series(win, "rew")
    rew_f = _finite(rew)
    rew_all_f = _finite(rew_all)
    scale = max(1.0, _med([abs(x) for x in rew_all_f]) if rew_all_f else 1.0)
    last = rew_f[-1] if rew_f else NAN
    first = rew_f[0] if rew_f else NAN
    early = rew_all_f[:3]
    out["rew_last"] = last
    out["rew_change_k"] = _safe_div(last - first, scale)
    out["rew_drop_from_max"] = _safe_div(max(rew_all_f) - last, scale) if rew_all_f else NAN
    out["rew_slope_k"] = _safe_div(_slope(rew), scale)
    diffs = [b - a for a, b in zip(rew_f, rew_f[1:])]
    out["rew_noise_k"] = _safe_div(_std(diffs), scale)
    out["rew_gain_vs_early"] = _safe_div(last - (sum(early) / len(early)), scale) if early else NAN

    eplen_all = _finite(series(rows, "eplen"))
    out["eplen_ratio"] = _safe_div(eplen_all[-1], max(eplen_all)) if eplen_all else NAN

    kl = series(win, "kl")
    kl_all = series(rows, "kl")
    kl_f = _finite(kl)
    out["kl_last"] = kl_f[-1] if kl_f else NAN
    out["kl_max_k"] = max(kl_f) if kl_f else NAN
    out["kl_ratio"] = _safe_div(out["kl_last"], _med(kl_all[:-1]) if len(kl_all) > 1 else NAN)

    clip_f = _finite(series(win, "clip"))
    out["clip_last"] = clip_f[-1] if clip_f else NAN

    ent_f = _finite(series(win, "ent"))
    out["ent_last"] = ent_f[-1] if ent_f else NAN
    out["ent_change_k"] = (ent_f[-1] - ent_f[0]) if len(ent_f) >= 2 else NAN

    vl_all = series(rows, "vloss")
    vl_f = _finite(vl_all)
    vl_last = vl_f[-1] if vl_f else NAN
    out["vloss_log"] = math.log10(abs(vl_last) + 1e-8) if math.isfinite(vl_last) else NAN
    out["vloss_ratio"] = _safe_div(vl_last, _med(vl_all[:-1]) if len(vl_all) > 1 else NAN)

    ev_f = _finite(series(win, "ev"))
    out["ev_last"] = ev_f[-1] if ev_f else NAN

    std_all = _finite(series(rows, "std"))
    out["std_last"] = std_all[-1] if std_all else NAN
    out["std_ratio"] = _safe_div(std_all[-1], std_all[0]) if std_all else NAN

    fps_all = series(rows, "fps")
    fps_f = _finite(fps_all)
    out["fps_ratio"] = _safe_div(fps_f[-1], _med(fps_all)) if fps_f else NAN

    lr_all = _finite(series(rows, "lr"))
    out["lr_ratio"] = _safe_div(lr_all[-1], lr_all[0]) if lr_all else NAN

    nonfinite = 0
    for r in win:
        for key in KEYS.values():
            v = r.get(key)
            if isinstance(v, float) and not math.isfinite(v):
                nonfinite += 1
    out["nonfinite_k"] = float(nonfinite)
    step = rows[-1].get("step", 0) or 0
    out["progress_log10"] = math.log10(float(step) + 1.0)

    # Clamp absurd magnitudes so a single exploded value cannot dominate; inf -> NaN.
    for name, v in out.items():
        out[name] = max(-1e6, min(1e6, v)) if math.isfinite(v) else NAN
    return out


def as_vector(feats: dict[str, float]) -> list[float]:
    return [feats.get(n, NAN) for n in FEATURE_NAMES]
