"""Builds the live detectors for a watcher profile from Settings, once per process.

Each detector is built on first use and on its own. The plain rules (NaN, silent log, dead
process) must never depend on a model: in the first real Temporal test on a 16 GB laptop,
TabPFN could not load ("The paging file is too small", Windows error 1455), and because every
activity built the whole profile up front, the crash/stall rules were blocked as well. Now a
model that fails to load only takes down its own check; the rules keep watching.
"""

from __future__ import annotations

import time
from pathlib import Path

from .config import Settings
from .detectors import PROFILES, observe
from .detectors.base import Signal


class WatchContext:
    def __init__(self, settings: Settings, profile: str = "nightwatch"):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        self.settings = settings
        self.profile = profile
        self.names = PROFILES[profile]
        # The local model also writes the morning summary, so every profile gets a client (cheap).
        from .llm import OllamaClient

        self.client = OllamaClient(settings.ollama_url, keep_alive=settings.ollama_keep_alive)
        self.judge = None
        self.tabpfn_model = None
        self._built: dict[str, object] = {}

    # ------------------------------------------------------------------ building
    def _build(self, name: str):
        s = self.settings
        if name == "rules":
            from .detectors import Rules

            return Rules()
        if name == "thresholds":
            from .detectors import Thresholds

            return Thresholds()
        if name == "vision":
            from .detectors.vision import VisionDetector, make_judge

            self.judge = make_judge(s, self.client)
            return VisionDetector(self.judge)
        if name == "tabpfn":
            from .detectors.tabpfn_detector import TabPFNDetector, WindowModel

            table = Path(s.tabpfn_train_table)
            if not table.is_file():
                raise FileNotFoundError(
                    f"TabPFN needs examples of past runs at {table}. Build it with: "
                    f"python -m nightwatch fit --runs runs/dataset --out {table}")
            t0 = time.time()
            # Fitted once per process and asked every few minutes: keep TabPFN's cache.
            self.tabpfn_model = WindowModel.from_table(table, s.tabpfn_backend, s.tabpfn_version,
                                                       fit_mode=s.tabpfn_fit_mode or None,
                                                       n_estimators=s.tabpfn_estimators or None)
            print(f"[watch] TabPFN backend: {self.tabpfn_model.backend_name} "
                  f"(fitted in {time.time() - t0:.0f} s)", flush=True)
            return TabPFNDetector(model=self.tabpfn_model)
        raise ValueError(f"unknown detector {name!r}")

    def detector(self, name: str):
        """Build on first use. Raises if it cannot be built (the caller decides what that means)."""
        if name not in self._built:
            self._built[name] = self._build(name)
        return self._built[name]

    def detectors_named(self, *names: str) -> list:
        return [self.detector(n) for n in self.names if n in names]

    @property
    def detectors(self) -> list:
        return self.detectors_named(*self.names)

    def warm_up(self) -> list[str]:
        """Build every detector now, at bedtime, so a model that cannot load (no licence, no
        memory, not pulled) shows up while you are still awake. Returns the problems found."""
        problems = []
        for name in self.names:
            try:
                self.detector(name)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
        return problems

    # ------------------------------------------------------------------ checking
    def check(self, run_dir: str, now: float | None = None) -> tuple[list[Signal], str, int]:
        """One check with every detector of the profile. A detector that cannot be built or
        crashes becomes a morning note; the others still run (used by `watch` and `check`)."""
        obs = observe(run_dir, now=now, live=True)
        signals: list[Signal] = []
        for name in self.names:
            try:
                signals.extend(self.detector(name).check(obs))
            except Exception as exc:  # noqa: BLE001
                signals.append(Signal(name, f"{name}_failed", "note", 0.0,
                                      f"{name} could not run: {type(exc).__name__}: {str(exc)[:160]}",
                                      obs.last_step, obs.now))
        return signals, obs.status, obs.last_step


_CACHE: dict[str, WatchContext] = {}


def get_context(profile: str, settings: Settings | None = None) -> WatchContext:
    if profile not in _CACHE:
        _CACHE[profile] = WatchContext(settings or Settings.from_env(), profile)
    return _CACHE[profile]
