"""Activities do the real work (read files, run TabPFN, call Gemma, send alerts). They can
fail and be retried; the workflow that calls them cannot lose its place."""

from __future__ import annotations

import time
from pathlib import Path

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .. import tracing
from ..config import Settings
from ..detectors import observe
from ..notify import Notifier
from ..report import write_morning_report
from ..runtime import get_context
from .shared import AlertInput, CheckInput, CheckOutput, ReportInput, ReportOutput

_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        tracing.init(_settings.sentry_dsn)
    return _settings


def _run(inp: CheckInput, *names: str) -> CheckOutput:
    ctx = get_context(inp.profile, settings())
    obs = observe(inp.run_dir, live=True)
    signals = []
    with tracing.transaction(f"{'+'.join(names)} {Path(inp.run_dir).name}"):
        for d in ctx.detectors_named(*names):
            signals.extend(d.check(obs))
    return CheckOutput(signals=signals, status=obs.status, last_step=obs.last_step)


@activity.defn(name="check_rules")
def check_rules(inp: CheckInput) -> CheckOutput:
    return _run(inp, "rules")


@activity.defn(name="check_thresholds")
def check_thresholds(inp: CheckInput) -> CheckOutput:
    return _run(inp, "thresholds")


@activity.defn(name="check_metrics")
def check_metrics(inp: CheckInput) -> CheckOutput:
    return _run(inp, "tabpfn")


@activity.defn(name="check_robot")
def check_robot(inp: CheckInput) -> CheckOutput:
    from ..llm import ModelMissing

    activity.heartbeat("looking at the robot")
    try:
        return _run(inp, "vision")
    except ModelMissing as exc:
        raise ApplicationError(str(exc), type="ModelMissing", non_retryable=True) from exc


@activity.defn(name="send_alert")
def send_alert(inp: AlertInput) -> list[str]:
    image = None
    if inp.image:
        p = Path(inp.image)
        image = str(p if p.is_absolute() else Path(inp.run_dir) / p)
    return Notifier(settings()).send(inp.title, inp.message, urgent=inp.urgent, image=image)


@activity.defn(name="write_report")
def write_report(inp: ReportInput) -> ReportOutput:
    s = settings()
    ctx = get_context(inp.profile, s)
    rep = write_morning_report(inp.run_dir, inp.state, client=ctx.client, text_model=s.text_model,
                               watch_started=inp.started_ts, watch_ended=inp.ended_ts or time.time(),
                               profile=inp.profile)
    return ReportOutput(markdown=rep["markdown"], summary=rep["summary"], summary_source=rep["summary_source"],
                        gif=rep.get("gif"), chart=rep.get("chart"))


ALL = [check_rules, check_thresholds, check_metrics, check_robot, send_alert, write_report]
