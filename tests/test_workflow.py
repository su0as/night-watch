"""The Temporal workflow over a simulated 8-hour night, using Temporal's time-skipping test
server (the night passes in seconds). Activities are fakes; the real ones are tested above."""

import asyncio

import pytest

try:
    from temporalio import activity
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from nightwatch.detectors.base import Signal
    from nightwatch.temporal_app.shared import AlertInput, CheckInput, CheckOutput, ReportInput, ReportOutput, WatchParams
    from nightwatch.temporal_app.workflows import NightWatch
except Exception as exc:  # noqa: BLE001  temporalio missing or its native bridge blocked
    pytest.skip(f"temporalio not usable here: {exc}", allow_module_level=True)

CALLS: dict = {}


@activity.defn(name="check_rules")
async def fake_rules(inp: CheckInput) -> CheckOutput:
    CALLS["checks"] = CALLS.get("checks", 0) + 1
    if CALLS["checks"] in (30, 31, 32):  # the trainer stalls around 02:30 and stays stalled
        return CheckOutput(signals=[Signal("rules", "stalled", "wake", 1.0, "No new metrics for 12 min.", 1)],
                           status="running", last_step=1)
    return CheckOutput(status="running", last_step=CALLS["checks"])


@activity.defn(name="check_metrics")
async def fake_metrics(inp: CheckInput) -> CheckOutput:
    return CheckOutput()


@activity.defn(name="check_thresholds")
async def fake_thresholds(inp: CheckInput) -> CheckOutput:
    CALLS["thresholds"] = CALLS.get("thresholds", 0) + 1
    return CheckOutput()


@activity.defn(name="check_robot")
async def fake_robot(inp: CheckInput) -> CheckOutput:
    CALLS["robot"] = CALLS.get("robot", 0) + 1
    if CALLS["robot"] == 5:
        raise RuntimeError("Ollama is loading the model")  # retried by Temporal
    return CheckOutput()


@activity.defn(name="send_alert")
async def fake_alert(inp: AlertInput) -> list[str]:
    CALLS.setdefault("alerts", []).append(inp.title)
    return ["test"]


@activity.defn(name="write_report")
async def fake_report(inp: ReportInput) -> ReportOutput:
    return ReportOutput(markdown="report.md", summary=f"{len(inp.state.wakes)} wake(s)", summary_source="template")


def test_eight_hour_night_with_continue_as_new():
    CALLS.clear()

    async def go():
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(env.client, task_queue="nw-test", workflows=[NightWatch],
                              activities=[fake_rules, fake_thresholds, fake_metrics, fake_robot, fake_alert, fake_report]):
                now = (await env.get_current_time()).timestamp()
                p = WatchParams(run_dir="/tmp/run", profile="nightwatch", until_ts=now + 8 * 3600,
                                check_every_s=300, checks_per_history=20)
                return await env.client.execute_workflow(NightWatch.run, p, id="nw-test-1", task_queue="nw-test")

    result = asyncio.run(go())
    assert result == "report.md"
    assert 90 <= CALLS["checks"] <= 100                    # one check every 5 min for 8 h
    assert CALLS["thresholds"] == CALLS["checks"]          # the cheap alerts run on every check too
    stalls = [a for a in CALLS["alerts"] if "stalled" in a]
    assert len(stalls) == 1                                # snoozed after the first wake, across continue-as-new
    assert CALLS["alerts"][-1] == "Night Watch: morning report"
