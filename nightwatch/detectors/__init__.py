"""Detectors and the watcher profiles compared in the evaluation.

    thresholds  rules + classic alert thresholds  (what most people run today)
    tabpfn      rules + TabPFN on training metrics
    vision      rules + Gemma looking at the robot
    nightwatch  rules + thresholds + TabPFN + Gemma   (the full thing: the cheap alerts stay on)
"""

from __future__ import annotations

from .base import Detector, Observation, Signal, observe
from .rules import Rules
from .threshold import Thresholds

PROFILES: dict[str, list[str]] = {
    "rules": ["rules"],
    "thresholds": ["rules", "thresholds"],
    "tabpfn": ["rules", "tabpfn"],
    "vision": ["rules", "vision"],
    "nightwatch": ["rules", "thresholds", "tabpfn", "vision"],
}


def build_detectors(names: list[str], *, tabpfn_model=None, tabpfn_lookup=None, vision_judge=None,
                    rules_kwargs: dict | None = None) -> list:
    out: list = []
    for name in names:
        if name == "rules":
            out.append(Rules(**(rules_kwargs or {})))
        elif name == "thresholds":
            out.append(Thresholds())
        elif name == "tabpfn":
            from .tabpfn_detector import TabPFNDetector

            out.append(TabPFNDetector(model=tabpfn_model, lookup=tabpfn_lookup))
        elif name == "vision":
            from .vision import VisionDetector

            if vision_judge is None:
                raise ValueError("vision detector needs a VisionJudge")
            out.append(VisionDetector(vision_judge))
        else:
            raise ValueError(f"unknown detector {name!r}")
    return out


def run_detectors(obs: Observation, detectors: list) -> list[Signal]:
    signals: list[Signal] = []
    for d in detectors:
        signals.extend(d.check(obs))
    return signals


__all__ = ["Detector", "Observation", "Signal", "observe", "PROFILES", "build_detectors", "run_detectors",
           "Rules", "Thresholds"]
