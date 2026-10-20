"""TabPFN reads the training metrics and says how much they look like runs that went wrong.

The training data is a plain CSV: one row per log window from past runs, with a label for
"something was already broken at this point". TabPFN does in-context learning, so "fitting"
is just handing it that table; there is no training loop to babysit (which would be ironic).

Backends, in order of preference:
    local   `tabpfn` package, open weights on your own machine (TabPFN-3.5 by default)
    client  `tabpfn-client`, Prior Labs' hosted API
    sklearn HistGradientBoosting, only as a fallback so the code runs without a licence.
            The backend name is printed on every alert and in every report, so the
            fallback can never be mistaken for TabPFN.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

from ..features import FEATURE_NAMES, as_vector, window_features
from .base import Observation, Signal

VERSION_ALIASES = {
    "v3.5": "V3_5", "3.5": "V3_5",
    "v3.5-fast": "V3_5_FAST", "fast": "V3_5_FAST",
    "v3": "V3", "v2.6": "V2_6", "v2.5": "V2_5", "v2": "V2",
}


def make_classifier(backend: str = "auto", version: str = "v3.5", fit_mode: str | None = None,
                    n_estimators: int | None = None):
    """Return (estimator, backend_name).

    fit_mode="fit_with_cache" makes TabPFN keep its attention cache of the training table after
    fit(), so each later predict() only processes the new rows. On a laptop CPU that turns a
    check from "re-read every past run" into well under a second, which is what a watcher that
    predicts every 5 minutes against the same table wants.
    """
    errors: list[str] = []
    overrides: dict = {}
    if fit_mode:
        overrides["fit_mode"] = fit_mode
    if n_estimators:
        overrides["n_estimators"] = int(n_estimators)
    if backend in ("auto", "local"):
        try:
            from tabpfn import TabPFNClassifier

            name = VERSION_ALIASES.get(version.lower(), "V3_5")
            try:
                from tabpfn.constants import ModelVersion

                mv = getattr(ModelVersion, name)
                clf = TabPFNClassifier.create_default_for_version(mv, **overrides)
            except Exception:  # noqa: BLE001  older/newer package layouts
                clf = TabPFNClassifier(**overrides)
            return clf, f"tabpfn-local-{version}"
        except ImportError as exc:
            errors.append(f"local: {exc}")
            if backend == "local":
                raise
    if backend in ("auto", "client"):
        try:
            from tabpfn_client import TabPFNClassifier as ClientClassifier

            return ClientClassifier(), "tabpfn-client"
        except ImportError as exc:
            errors.append(f"client: {exc}")
            if backend == "client":
                raise
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200), "sklearn-fallback"


class WindowModel:
    """A fitted 'does this window look like trouble?' classifier, plus a cheap explanation."""

    def __init__(self, backend: str = "auto", version: str = "v3.5", fit_mode: str | None = None,
                 n_estimators: int | None = None):
        self.backend = backend
        self.version = version
        self.fit_mode = fit_mode
        self.n_estimators = n_estimators
        self.backend_name = "unfitted"
        self._clf = None
        self._constant: float | None = None
        self._healthy_mean: list[float] = []
        self._healthy_std: list[float] = []

    # -- fitting
    def fit(self, X: list[list[float]], y: list[int]) -> "WindowModel":
        import numpy as np

        if not X:
            raise ValueError("no training windows")
        classes = set(int(v) for v in y)
        self._fit_explainer(X, y)
        if len(classes) < 2:
            self._constant = float(classes.pop())
            self.backend_name = "constant"
            return self
        clf, name = make_classifier(self.backend, self.version, self.fit_mode, self.n_estimators)
        clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        self._clf, self.backend_name = clf, name
        return self

    def _fit_explainer(self, X: list[list[float]], y: list[int]) -> None:
        healthy = [row for row, lab in zip(X, y) if lab == 0] or X
        cols = list(zip(*healthy))
        self._healthy_mean, self._healthy_std = [], []
        for col in cols:
            f = [v for v in col if math.isfinite(v)]
            m = sum(f) / len(f) if f else 0.0
            s = math.sqrt(sum((v - m) ** 2 for v in f) / max(len(f) - 1, 1)) if len(f) > 1 else 1.0
            self._healthy_mean.append(m)
            self._healthy_std.append(s if s > 1e-9 else 1.0)

    # -- prediction
    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if self._constant is not None:
            return [self._constant] * len(X)
        if self._clf is None:
            raise RuntimeError("model not fitted")
        import numpy as np

        proba = self._clf.predict_proba(np.asarray(X, dtype=float))
        classes = list(getattr(self._clf, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else len(classes) - 1
        return [float(p[idx]) for p in proba]

    def explain(self, x: list[float], top: int = 2) -> list[tuple[str, float]]:
        """Which features are furthest from healthy windows (z-score). Not TabPFN's reasoning,
        just a readable hint for the alert text."""
        if not self._healthy_mean:
            return []
        zs = []
        for name, v, m, s in zip(FEATURE_NAMES, x, self._healthy_mean, self._healthy_std):
            if math.isfinite(v):
                zs.append((name, (v - m) / s))
        zs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return zs[:top]

    # -- persistence
    @staticmethod
    def load_table(path: str | Path) -> tuple[list[list[float]], list[int], list[dict]]:
        X, y, meta = [], [], []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                X.append([float(row[n]) if row[n] not in ("", "nan", "NaN") else math.nan for n in FEATURE_NAMES])
                y.append(int(float(row["label"])))
                meta.append({"run_id": row.get("run_id", ""), "step": int(float(row.get("step", 0) or 0))})
        return X, y, meta

    @classmethod
    def from_table(cls, path: str | Path, backend: str = "auto", version: str = "v3.5",
                   exclude_run: str | None = None, fit_mode: str | None = None,
                   n_estimators: int | None = None) -> "WindowModel":
        X, y, meta = cls.load_table(path)
        if exclude_run:
            keep = [i for i, m in enumerate(meta) if m["run_id"] != exclude_run]
            X, y = [X[i] for i in keep], [y[i] for i in keep]
        return cls(backend, version, fit_mode, n_estimators).fit(X, y)


ProbLookup = Callable[[str, int], "float | None"]


class TabPFNDetector:
    name = "tabpfn"

    def __init__(self, model: WindowModel | None = None, wake_p: float = 0.8, note_p: float = 0.5,
                 consecutive: int = 2, k: int = 8, lookup: ProbLookup | None = None,
                 min_rows: int = 6):
        if model is None and lookup is None:
            raise ValueError("TabPFNDetector needs a fitted model or a probability lookup")
        self.model = model
        self.lookup = lookup
        self.wake_p = wake_p
        self.note_p = note_p
        self.consecutive = consecutive
        self.k = k
        self.min_rows = min_rows

    def _probs(self, obs: Observation) -> tuple[list[float], list[float]]:
        rows = obs.metrics
        n = len(rows)
        ends = list(range(max(self.min_rows, n - self.consecutive + 1), n + 1))
        if self.lookup is not None:
            ps = []
            for e in ends:
                p = self.lookup(obs.run_id, int(rows[e - 1].get("step", 0)))
                if p is not None:
                    ps.append(p)
            last_x = as_vector(window_features(rows, self.k))
            return ps, last_x
        xs = [as_vector(window_features(rows[:e], self.k)) for e in ends]
        ps = self.model.predict_proba(xs) if xs else []
        return ps, xs[-1] if xs else []

    def check(self, obs: Observation) -> list[Signal]:
        if len(obs.metrics) < self.min_rows:
            return []
        probs, last_x = self._probs(obs)
        if not probs:
            return []
        p = probs[-1]
        step = obs.last_step
        backend = self.model.backend_name if self.model else "lookup"
        hint = ""
        if self.model and last_x:
            ex = self.model.explain(last_x)
            if ex:
                hint = " Furthest from healthy: " + ", ".join(f"{n} ({z:+.1f} sd)" for n, z in ex) + "."
        if len(probs) >= self.consecutive and all(q >= self.wake_p for q in probs[-self.consecutive:]):
            return [Signal("tabpfn", "metrics_anomaly", "wake", p,
                           f"Training metrics look like past runs that broke (p={p:.2f}).{hint}",
                           step, obs.now, {"backend": backend})]
        if p >= self.note_p:
            return [Signal("tabpfn", "metrics_anomaly", "note", p,
                           f"Metrics drifting toward past failures (p={p:.2f}).{hint}",
                           step, obs.now, {"backend": backend})]
        return []
