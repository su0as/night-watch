"""Scoring a watcher on one run: did it wake you, how fast, and how often for nothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from ..policy import Alert
from .groundtruth import Failure, RunSpan


@dataclass
class RunScore:
    run_id: str
    profile: str
    failure_kind: str            # "none" for a clean run
    caught: bool | None          # None when there was nothing to catch
    minutes_to_wake: float | None
    steps_to_wake: int | None
    first_wake_source: str | None
    false_wakes: int
    hours: float
    wake_kinds: list[str] = field(default_factory=list)


def score_run(run_id: str, profile: str, failures: list[Failure], wakes: list[Alert], span: RunSpan
              ) -> RunScore:
    hours = max(span.end_t - span.start_t, 0.0) / 3600.0
    kinds = [f"{w.source}/{w.kind}" for w in wakes]
    if not failures:
        return RunScore(run_id, profile, "none", None, None, None, None, len(wakes), hours, kinds)
    onset = failures[0]
    before = [w for w in wakes if w.t < onset.onset_t]
    after = [w for w in wakes if w.t >= onset.onset_t]
    first = after[0] if after else None
    return RunScore(
        run_id, profile, onset.kind, first is not None,
        round((first.t - onset.onset_t) / 60.0, 1) if first else None,
        (first.step - onset.onset_step) if first and first.step >= 0 else None,
        f"{first.source}/{first.kind}" if first else None,
        len(before), hours, kinds,
    )


@dataclass
class ProfileSummary:
    profile: str
    failures: int
    caught: int
    median_minutes: float | None
    false_wakes: int
    clean_runs: int
    false_wakes_clean: int
    hours: float

    @property
    def false_per_8h(self) -> float:
        return round(self.false_wakes / self.hours * 8, 2) if self.hours > 0 else 0.0


def summarise(scores: list[RunScore]) -> list[ProfileSummary]:
    out = []
    for profile in dict.fromkeys(s.profile for s in scores):
        ss = [s for s in scores if s.profile == profile]
        with_fail = [s for s in ss if s.caught is not None]
        caught = [s for s in with_fail if s.caught]
        mins = [s.minutes_to_wake for s in caught if s.minutes_to_wake is not None]
        clean = [s for s in ss if s.caught is None]
        out.append(ProfileSummary(
            profile=profile, failures=len(with_fail), caught=len(caught),
            median_minutes=round(median(mins), 1) if mins else None,
            false_wakes=sum(s.false_wakes for s in ss), clean_runs=len(clean),
            false_wakes_clean=sum(s.false_wakes for s in clean), hours=round(sum(s.hours for s in ss), 2)))
    return out
