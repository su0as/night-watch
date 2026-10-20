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

    print("\nOllama + Gemma")
    try:
        import requests

        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        _ok(f"Ollama at {settings.ollama_url} ({len(names)} models)")
        for model in {settings.vision_model, settings.text_model}:
            if any(n == model or n.startswith(model + ":") or n.split(":")[0] == model for n in names):
                _ok(f"{model} is pulled")
            else:
                failures += 1
                _bad(f"{model} not pulled", f"ollama pull {model}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        _bad(f"Ollama not reachable: {exc}", "install from https://ollama.com and start it")

    print("\nTabPFN")
    try:
        from .detectors.tabpfn_detector import make_classifier

        _, name = make_classifier(settings.tabpfn_backend, settings.tabpfn_version)
        if name.startswith("sklearn"):
            _warn("TabPFN not importable; the sklearn fallback would be used. Install tabpfn for real results.")
        else:
            _ok(f"backend {name} (first fit opens a browser to accept the licence, or set TABPFN_TOKEN)")
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
