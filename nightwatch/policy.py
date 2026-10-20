"""Decide whether a finding is worth waking a human.

Pure and deterministic: the same function runs inside the Temporal workflow (which must
be deterministic), in the plain watcher loop, and in the offline evaluation.

Rules of politeness:
- a "wake" signal wakes you, unless the same kind already woke you within the snooze window;
- after `max_wakes` wake-ups in one night, everything else goes to the morning report;
- "note" signals are collected for the morning report, deduplicated by kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detectors.base import Signal


@dataclass
class PolicyConfig:
    max_wakes: int = 3
    snooze_s: float = 3600.0
    max_notes: int = 50


@dataclass
class Alert:
    level: str            # "wake" | "note"
    kind: str
    source: str
    message: str
    step: int
    t: float
    score: float = 0.0
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass
class PolicyState:
    wakes: list[Alert] = field(default_factory=list)
    notes: list[Alert] = field(default_factory=list)
    last_wake_by_kind: dict[str, float] = field(default_factory=dict)
    last_note_by_kind: dict[str, float] = field(default_factory=dict)
    suppressed: int = 0
    checks: int = 0


def _to_alert(sig: Signal, level: str, now: float) -> Alert:
    return Alert(level=level, kind=sig.kind, source=sig.source, message=sig.message, step=sig.step,
                 t=sig.t or now, score=float(sig.score), evidence=dict(sig.evidence or {}))


def decide(state: PolicyState, signals: list[Signal], now: float, cfg: PolicyConfig | None = None
           ) -> tuple[list[Alert], PolicyState]:
    """Returns (alerts to send right now, new state). Only 'wake' alerts are returned for
    sending; notes go into state for the morning report."""
    cfg = cfg or PolicyConfig()
    st = PolicyState(wakes=list(state.wakes), notes=list(state.notes),
                     last_wake_by_kind=dict(state.last_wake_by_kind),
                     last_note_by_kind=dict(state.last_note_by_kind),
                     suppressed=state.suppressed, checks=state.checks + 1)
    to_send: list[Alert] = []
    # Strongest first, so if we can only wake once, it is for the most serious thing.
    for sig in sorted(signals, key=lambda s: (s.level != "wake", -float(s.score))):
        if sig.level == "wake":
            last = st.last_wake_by_kind.get(sig.kind)
            snoozed = last is not None and now - last < cfg.snooze_s
            if not snoozed and len(st.wakes) < cfg.max_wakes:
                alert = _to_alert(sig, "wake", now)
                st.wakes.append(alert)
                st.last_wake_by_kind[sig.kind] = now
                to_send.append(alert)
                continue
            st.suppressed += 1
            sig_level = "note"
        else:
            sig_level = sig.level
        if sig_level == "note":
            last_note = st.last_note_by_kind.get(sig.kind)
            if last_note is None or now - last_note >= cfg.snooze_s:
                if len(st.notes) < cfg.max_notes:
                    st.notes.append(_to_alert(sig, "note", now))
                st.last_note_by_kind[sig.kind] = now
    return to_send, st
