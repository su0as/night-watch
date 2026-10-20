"""python -m nightwatch doctor: check every moving part before you trust it with a night."""

from __future__ import annotations

import importlib
import platform
import sys

from .config import Settings


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _bad(msg: str, fix: str = "") -> None:
    print(f"  [FAIL] {msg}" + (f"\n         fix: {fix}" if fix else ""))


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def run(settings: Settings, render: bool = True) -> int:
    failures = 0
    print(f"Night Watch doctor on {platform.system()} {platform.release()}, Python {sys.version.split()[0]}")

    print("\nPython packages")
    for mod, extra in [("numpy", ""), ("PIL", ""), ("requests", ""), ("psutil", ""),
                       ("gymnasium", "train"), ("mujoco", "train"), ("stable_baselines3", "train"),
                       ("tabpfn", "tabpfn"), ("temporalio", "temporal"), ("matplotlib", "plots")]:
        try:
            m = importlib.import_module(mod)
            _ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _bad(f"{mod}: {exc}", f'pip install -e ".[{extra or "all"}]"')

    if render:
        print("\nMuJoCo rendering (needed for Gemma to see the robot)")
        try:
            import gymnasium as gym

            env = gym.make("HalfCheetah-v5", render_mode="rgb_array", width=160, height=120)
            env.reset(seed=0)
            frame = env.render()
            env.close()
            if frame is not None and getattr(frame, "shape", (0,))[0] == 120:
                _ok(f"HalfCheetah renders, frame {frame.shape}")
            else:
                failures += 1
                _bad("render returned nothing")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _bad(f"render failed: {exc}", "update GPU drivers; on headless Linux set MUJOCO_GL=egl")

    print("\nThis machine overnight")
    _machine(settings)

    print("\nOllama + Gemma")
    try:
        import requests

        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        models = r.json().get("models", [])
        names = [m.get("name", "") for m in models]
        _ok(f"Ollama at {settings.ollama_url} ({len(names)} models)")
        for model in {settings.vision_model, settings.text_model}:
            full = model if ":" in model else model + ":latest"
            if full in names or model in names:
                size = next((m.get("size", 0) for m in models if m.get("name") in (model, full)), 0)
                _ok(f"{model} is pulled ({size / 1e9:.1f} GB)")
            else:
                failures += 1
                _bad(f"{model} not pulled", f"ollama pull {model}")
        mode = "thinks before answering" if settings.vision_think else "answers straight away"
        _ok(f"vision judge {mode}, {settings.vision_snapshots} snapshot(s) at up to {settings.vision_max_side} px; "
            "check its eyesight with: python -m nightwatch eyetest")
        if settings.vision_max_side < 800 or settings.vision_snapshots == 0:
            _warn("small pictures blind Gemma: Ollama gives it ~1 image token per 48x48 px. In our eyetest a "
                  "320x240 snapshot got the same answer whatever the robot was doing")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        _bad(f"Ollama not reachable: {exc}", "install from https://ollama.com and start it (ollama serve)")

    print("\nTabPFN")
    try:
        from .detectors.tabpfn_detector import make_classifier

        _, name = make_classifier(settings.tabpfn_backend, settings.tabpfn_version)
        if name.startswith("sklearn"):
            _warn("TabPFN not importable; the sklearn fallback would be used. Install tabpfn for real results.")
        else:
            _ok(f"backend {name}")
            if name.startswith("tabpfn-local"):
                _tabpfn_licence()
    except Exception as exc:  # noqa: BLE001
        failures += 1
        _bad(f"{exc}")

    print("\nTemporal")
    try:
        import asyncio

        from temporalio.client import Client

        async def _ping() -> None:
            await Client.connect(settings.temporal_address)

        asyncio.run(asyncio.wait_for(_ping(), timeout=5))
        _ok(f"server at {settings.temporal_address}")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Temporal server not reachable ({type(exc).__name__}). Start it with: "
              "temporal server start-dev --db-filename nightwatch.db")

    print("\nPhone alerts")
    if settings.ntfy_topic:
        _ok(f"ntfy topic {settings.ntfy_topic}")
    elif settings.telegram_token:
        _ok("telegram configured")
    else:
        _warn("no phone channel set (NW_NTFY_TOPIC or NW_TELEGRAM_*); alerts go to the console only")

    print("\n" + ("All good." if failures == 0 else f"{failures} problem(s) above."))
    return failures


def _tabpfn_licence() -> None:
    import os
    from pathlib import Path

    token = os.environ.get("TABPFN_TOKEN", "").strip()
    cached = any(p.is_file() for p in (Path.home() / ".cache" / "tabpfn" / "auth_token",
                                       Path.home() / ".tabpfn" / "token"))
    if token:
        _ok("licence token set (TABPFN_TOKEN)")
    elif cached:
        _ok("licence token cached from an earlier login")
    elif sys.platform == "win32":
        _warn("no TabPFN licence token. On Windows, TabPFN's terminal login prompt crashes (it uses "
              "select() on stdin, which Windows does not support). Accept the licence at "
              "https://ux.priorlabs.ai, create an API key and put TABPFN_TOKEN=... in .env")
    else:
        _warn("no TabPFN licence token yet: the first fit opens a browser, or set TABPFN_TOKEN in .env")


def _windows_commit_free_gb() -> float | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPageFile / 1e9
    except Exception:  # noqa: BLE001
        return None


def _machine(settings: Settings) -> None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        free_gb, total_gb = vm.available / 1e9, vm.total / 1e9
        msg = f"{free_gb:.1f} GB of {total_gb:.0f} GB RAM free, {psutil.cpu_count(logical=False)} CPU cores"
        if free_gb < 8:
            _warn(msg + ". Gemma 4 E4B needs ~6 GB and each training run ~0.5 GB: close big apps at night")
        else:
            _ok(msg)
    except Exception:  # noqa: BLE001
        pass
    if sys.platform != "win32":
        return
    commit = _windows_commit_free_gb()
    if commit is not None:
        if commit < 4:
            _warn(f"only {commit:.1f} GB of commit (RAM + page file) left. Gemma, TabPFN and 3 training runs "
                  "can run Windows out of memory overnight. Close apps, or add a page file on a roomier drive "
                  "(System Properties > Advanced > Performance > Virtual memory)")
        else:
            _ok(f"{commit:.1f} GB of commit (RAM + page file) left")
    from .keepawake import windows_sleep_settings

    sl = windows_sleep_settings()
    ac = sl.get("sleep_ac_min")
    if ac:
        _warn(f"plugged in, Windows sleeps after {ac} min idle. Training and watching ask Windows to stay "
              "awake while they run, but setting 'When plugged in, put my device to sleep after: Never' is safer")
    elif ac == 0:
        _ok("plugged in, Windows never sleeps on its own")
    if sl.get("lid_ac") not in (None, 0):
        _warn("closing the lid while plugged in puts the laptop to sleep. Leave it open, or set "
              "'When I close the lid: Do nothing' (Control Panel > Power Options > lid)")
