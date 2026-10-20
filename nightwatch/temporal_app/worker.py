"""Run the worker:  python -m nightwatch worker
Start the watch:  python -m nightwatch start runs/my-run --until 07:30
Peek / stop:      python -m nightwatch status runs/my-run   |   python -m nightwatch stop runs/my-run
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker

from ..config import Settings
from .activities import ALL
from .shared import WatchParams
from .workflows import NightWatch


def workflow_id(run_dir: str) -> str:
    return f"nightwatch-{Path(run_dir).resolve().name}"


async def run_worker(settings: Settings) -> None:
    client = await Client.connect(settings.temporal_address)
    with ThreadPoolExecutor(max_workers=4) as pool:
        worker = Worker(client, task_queue=settings.task_queue, workflows=[NightWatch], activities=ALL,
                        activity_executor=pool)
        print(f"[worker] listening on {settings.temporal_address} queue={settings.task_queue}", flush=True)
        await worker.run()


async def start_watch(settings: Settings, run_dir: str, profile: str, until_ts: float, every_s: float) -> str:
    client = await Client.connect(settings.temporal_address)
    p = WatchParams(run_dir=str(Path(run_dir).resolve()), profile=profile, until_ts=until_ts,
                    check_every_s=every_s, max_wakes=settings.max_wakes, snooze_s=settings.snooze_min * 60)
    handle = await client.start_workflow(NightWatch.run, p, id=workflow_id(run_dir), task_queue=settings.task_queue)
    return handle.id


async def query_status(settings: Settings, run_dir: str) -> dict:
    client = await Client.connect(settings.temporal_address)
    return await client.get_workflow_handle(workflow_id(run_dir)).query(NightWatch.status)


async def signal(settings: Settings, run_dir: str, name: str, *args) -> None:
    client = await Client.connect(settings.temporal_address)
    handle = client.get_workflow_handle(workflow_id(run_dir))
    if name == "stop":
        await handle.signal(NightWatch.stop)
    elif name == "snooze":
        await handle.signal(NightWatch.snooze, float(args[0]) if args else 60.0)


def main(settings: Settings) -> None:
    asyncio.run(run_worker(settings))
