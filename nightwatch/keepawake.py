"""Keep the computer awake while a run or a watch is going.

A laptop that falls asleep at 1 a.m. pauses the training *and* the watcher, and the night is
gone. Windows lets a program say "I'm busy, don't sleep" (SetThreadExecutionState, what video
players use); macOS has `caffeinate`. The screen can still turn off, which is fine.

Closing the lid may still put the machine to sleep, depending on your power settings
("When I close the lid: Do nothing"). That part is up to you; `doctor` warns about it.
Set NW_ALLOW_SLEEP=1 to turn this off.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from typing import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _disabled() -> bool:
    return os.environ.get("NW_ALLOW_SLEEP", "").strip().lower() in ("1", "true", "yes", "on")


@contextlib.contextmanager
def keep_awake() -> Iterator[bool]:
    """Yields True if the OS accepted the request. Never raises."""
    if _disabled():
        yield False
        return
    if sys.platform == "win32":
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            ok = bool(k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        except Exception:  # noqa: BLE001
            k32, ok = None, False
        try:
            yield ok
        finally:
            if k32 is not None:
                try:
                    k32.SetThreadExecutionState(ES_CONTINUOUS)
                except Exception:  # noqa: BLE001
                    pass
        return
    if sys.platform == "darwin":
        proc = None
        try:
            proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            proc = None
        try:
            yield proc is not None
        finally:
            if proc is not None:
                proc.terminate()
        return
    yield False


def windows_sleep_settings() -> dict[str, int | None]:
    """Minutes until sleep on AC/battery and the lid-close action on AC (0 nothing, 1 sleep,
    2 hibernate, 3 shut down). None if it cannot be read. Windows only."""
    out: dict[str, int | None] = {"sleep_ac_min": None, "sleep_dc_min": None, "lid_ac": None}
    if sys.platform != "win32":
        return out

    def _query(sub: str, setting: str) -> tuple[int | None, int | None]:
        try:
            txt = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", sub, setting],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            return None, None
        ac = dc = None
        for line in txt.splitlines():
            low = line.lower()
            if "current ac power setting index" in low:
                ac = int(line.rsplit(":", 1)[1].strip(), 16)
            elif "current dc power setting index" in low:
                dc = int(line.rsplit(":", 1)[1].strip(), 16)
        return ac, dc

    ac, dc = _query("SUB_SLEEP", "STANDBYIDLE")
    out["sleep_ac_min"] = None if ac is None else ac // 60
    out["sleep_dc_min"] = None if dc is None else dc // 60
    # Lid-close action has no alias on some machines; use its GUID. Hidden on many Modern
    # Standby laptops, in which case it stays None.
    lid_ac, _ = _query("SUB_BUTTONS", "5ca83367-6e45-459f-a27b-476b1d01c936")
    out["lid_ac"] = lid_ac
    return out
