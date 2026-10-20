"""When did each run actually go wrong?

Two sources, both hidden from the watchers:
- injected training faults: the exact step/time the fault fired (labels/fault_events.jsonl);
- behaviour failures: the first test episode where the simulator says the robot was flipped,
  spinning or exploded, sustained for `sustain` consecutive test episodes (labels/truth.jsonl).
  This also catches the robot discovering a bad trick on its own in a "healthy" run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..faults import TRAINING_FAULTS
from ..runlog import LabelReader, RunReader, read_jsonl
from ..truth import label_episode


@dataclass
class Failure:
    kind: str
    onset_step: int
    onset_t: float
    source: str  # "injected" | "behaviour"


@dataclass
class RunSpan:
    start_t: float
    end_t: float
    finished: bool


def behaviour_failures(truth_rows: list[dict], sustain: int = 2) -> list[Failure]:
    rows = [r for r in truth_rows if int(r.get("step", 0)) > 0]
    labels = [r.get("label") or label_episode(r) for r in rows]
    for i, lab in enumerate(labels):
        if lab == "ok":
            continue
        window = labels[i:i + sustain]
        sustained = len(window) == sustain and all(x != "ok" for x in window)
        if sustained or i == len(labels) - 1:
            r = rows[i]
            return [Failure(kind=lab, onset_step=int(r["step"]), onset_t=float(r["t"]), source="behaviour")]
    return []


def injected_failures(run_dir: str | Path) -> list[Failure]:
    lab = LabelReader(run_dir)
    fault = lab.fault()
    if fault.get("kind") not in TRAINING_FAULTS:
        return []
    events = read_jsonl(lab.paths.labels_dir / "fault_events.jsonl")
    if not events:
        return []  # the run ended before the fault fired
    ev = events[0]
    return [Failure(kind=str(fault["kind"]), onset_step=int(ev["step"]), onset_t=float(ev["t"]), source="injected")]


def failures_for_run(run_dir: str | Path, sustain: int = 2) -> list[Failure]:
    injected = injected_failures(run_dir)
    behaviour = behaviour_failures(LabelReader(run_dir).truth(), sustain=sustain)
    if injected and behaviour and behaviour[0].onset_t >= injected[0].onset_t:
        behaviour = []  # same incident: the injected fault caused the bad behaviour
    return sorted(injected + behaviour, key=lambda f: f.onset_t)


def run_span(run_dir: str | Path) -> RunSpan:
    reader = RunReader(run_dir)
    m = LabelReader(run_dir).full_manifest()
    rows = reader.metrics()
    frames = reader.frames()
    ts = [float(r["t"]) for r in rows] + [float(f["t"]) for f in frames]
    start = float(m.get("started_at") or (min(ts) if ts else 0.0))
    if m.get("status") in ("finished", "error") and m.get("ended_at"):
        return RunSpan(start, float(m["ended_at"]), True)
    return RunSpan(start, max(ts) if ts else start, False)
