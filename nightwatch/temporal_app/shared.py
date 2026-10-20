"""Data passed between the workflow and its activities. Plain dataclasses, so Temporal's
default JSON converter can store them in the workflow history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..detectors.base import Signal
from ..policy import PolicyState

WORKFLOW_NAME = "NightWatch"


@dataclass
class WatchParams:
    run_dir: str
    profile: str = "nightwatch"
    until_ts: float = 0.0
    check_every_s: float = 300.0
    max_wakes: int = 3
    snooze_s: float = 3600.0
    checks_per_history: int = 150
    started_ts: float = 0.0
    state: Optional[PolicyState] = None


@dataclass
class CheckInput:
    run_dir: str
    profile: str


@dataclass
class CheckOutput:
    signals: list[Signal] = field(default_factory=list)
    status: str = "running"
    last_step: int = 0


@dataclass
class AlertInput:
    run_dir: str
    title: str
    message: str
    urgent: bool = True
    image: Optional[str] = None


@dataclass
class ReportInput:
    run_dir: str
    profile: str
    state: PolicyState
    started_ts: float
    ended_ts: float


@dataclass
class ReportOutput:
    markdown: str
    summary: str
    summary_source: str
    gif: Optional[str] = None
    chart: Optional[str] = None
