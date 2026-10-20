"""Builds the live detectors for a watcher profile from Settings, once per process."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .detectors import PROFILES, build_detectors, observe, run_detectors
from .detectors.base import Signal


class WatchContext:
    def __init__(self, settings: Settings, profile: str = "nightwatch"):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        self.settings = settings
        self.profile = profile
        self.names = PROFILES[profile]
        self.client = None
        self.judge = None
        self.tabpfn_model = None
        # The local model also writes the morning summary, so every profile gets a client.
        from .llm import OllamaClient

        self.client = OllamaClient(settings.ollama_url)
        if "vision" in self.names:
            from .detectors.vision import VisionJudge

            self.judge = VisionJudge(self.client, settings.vision_model)
        if "tabpfn" in self.names:
            from .detectors.tabpfn_detector import WindowModel

            table = Path(settings.tabpfn_train_table)
            if not table.is_file():
                raise FileNotFoundError(
                    f"TabPFN needs examples of past runs at {table}. Build it with: "
                    f"python -m nightwatch fit --runs runs/dataset --out {table}")
            self.tabpfn_model = WindowModel.from_table(table, settings.tabpfn_backend, settings.tabpfn_version)
            print(f"[watch] TabPFN backend: {self.tabpfn_model.backend_name}", flush=True)
        self.detectors = build_detectors(self.names, tabpfn_model=self.tabpfn_model, vision_judge=self.judge)

    def check(self, run_dir: str, now: float | None = None) -> tuple[list[Signal], str, int]:
        obs = observe(run_dir, now=now, live=True)
        return run_detectors(obs, self.detectors), obs.status, obs.last_step

    def detectors_named(self, *names: str) -> list:
        return [d for d in self.detectors if d.name in names]


_CACHE: dict[str, WatchContext] = {}


def get_context(profile: str, settings: Settings | None = None) -> WatchContext:
    if profile not in _CACHE:
        _CACHE[profile] = WatchContext(settings or Settings.from_env(), profile)
    return _CACHE[profile]
