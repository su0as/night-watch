"""The overnight watch as a durable workflow.

The loop below looks like ordinary Python, but every `await` is recorded by Temporal. If the
worker process dies at 03:12 (laptop lid closed, Windows update, OOM), the workflow is still
"running" on the Temporal server. Restart the worker and it replays the history and carries on
from the last completed check, including how many times it has already woken you.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from ..detectors import PROFILES
    from ..detectors.base import Signal
    from ..policy import PolicyConfig, PolicyState, decide
    from .shared import (WORKFLOW_NAME, AlertInput, CheckInput, CheckOutput, ReportInput, ReportOutput,
                         WatchParams)

QUICK = RetryPolicy(initial_interval=timedelta(seconds=5), backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=2), maximum_attempts=5)
MODEL = RetryPolicy(initial_interval=timedelta(seconds=15), backoff_coefficient=2.0,
                    maximum_interval=timedelta(minutes=5), maximum_attempts=3,
                    non_retryable_error_types=["ModelMissing"])


@workflow.defn(name=WORKFLOW_NAME)
class NightWatch:
    def __init__(self) -> None:
        self._stop = False
        self._snooze_until = 0.0
        self._state = PolicyState()
        self._status = "starting"
        self._last_step = 0

    @workflow.signal
    def stop(self) -> None:
        """Wake up early and want the report now."""
        self._stop = True

    @workflow.signal
    def snooze(self, minutes: float) -> None:
        """Seen it, let me sleep: no wake-ups for N minutes (still noted for the morning)."""
        self._snooze_until = workflow.now().timestamp() + float(minutes) * 60

    @workflow.query
    def status(self) -> dict:
        return {"status": self._status, "last_step": self._last_step, "wakes": len(self._state.wakes),
                "notes": len(self._state.notes), "checks": self._state.checks,
                "snoozed_until": self._snooze_until}

    async def _check(self, name: str, inp: CheckInput, timeout_min: float, retry: RetryPolicy) -> CheckOutput:
        try:
            return await workflow.execute_activity(name, inp, result_type=CheckOutput,
                                                   start_to_close_timeout=timedelta(minutes=timeout_min),
                                                   retry_policy=retry)
        except ActivityError as err:
            cause = getattr(err, "cause", None) or err
            msg = str(getattr(cause, "message", None) or cause)[:200]
            return CheckOutput(signals=[Signal(name, f"{name}_failed", "note", 0.0,
                                               f"{name} failed after retries: {msg}")])

    @workflow.run
    async def run(self, p: WatchParams) -> str:
        self._state = p.state or PolicyState()
        started = p.started_ts or workflow.now().timestamp()
        cfg = PolicyConfig(max_wakes=p.max_wakes, snooze_s=p.snooze_s)
        names = PROFILES[p.profile]
        inp = CheckInput(run_dir=p.run_dir, profile=p.profile)
        checks_this_history = 0

        while not self._stop and workflow.now().timestamp() < p.until_ts:
            rules = await self._check("check_rules", inp, 2, QUICK)
            signals = list(rules.signals)
            self._status, self._last_step = rules.status, rules.last_step
            if "thresholds" in names:
                signals += (await self._check("check_thresholds", inp, 2, QUICK)).signals
            if "tabpfn" in names:
                signals += (await self._check("check_metrics", inp, 10, MODEL)).signals
            if "vision" in names:
                signals += (await self._check("check_robot", inp, 20, MODEL)).signals

            now = workflow.now().timestamp()
            if now < self._snooze_until:
                signals = [dataclasses.replace(s, level="note") if s.level == "wake" else s for s in signals]
            alerts, self._state = decide(self._state, signals, now, cfg)
            for a in alerts:
                await workflow.execute_activity(
                    "send_alert",
                    AlertInput(run_dir=p.run_dir, title=f"Night Watch: {a.kind.replace('_', ' ')}",
                               message=a.message, urgent=True, image=a.evidence.get("sheet")),
                    result_type=list, start_to_close_timeout=timedelta(minutes=2), retry_policy=QUICK)

            if self._status in ("finished", "error"):
                break
            checks_this_history += 1
            if checks_this_history >= p.checks_per_history:
                workflow.continue_as_new(dataclasses.replace(p, state=self._state, started_ts=started))
            try:
                await workflow.wait_condition(lambda: self._stop, timeout=timedelta(seconds=p.check_every_s))
            except asyncio.TimeoutError:
                pass

        rep = await workflow.execute_activity(
            "write_report",
            ReportInput(run_dir=p.run_dir, profile=p.profile, state=self._state, started_ts=started,
                        ended_ts=workflow.now().timestamp()),
            result_type=ReportOutput, start_to_close_timeout=timedelta(minutes=20), retry_policy=QUICK)
        await workflow.execute_activity(
            "send_alert",
            AlertInput(run_dir=p.run_dir, title="Night Watch: morning report", message=rep.summary,
                       urgent=False, image=rep.gif),
            result_type=list, start_to_close_timeout=timedelta(minutes=2), retry_policy=QUICK)
        return rep.markdown
