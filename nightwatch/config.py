"""Settings from environment variables, with an optional .env file (no extra dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments, existing env vars win."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool) -> bool:
    v = _env(name, "").lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


@dataclass
class Settings:
    runs_dir: str = "runs"
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "gemma4:e4b"
    text_model: str = "gemma4:e4b"
    vision_view: str = "head"
    vision_think: bool = False
    vision_max_tokens: int = 1500
    vision_snapshots: int = 2
    vision_reference: bool = False
    vision_max_side: int = 1024
    tabpfn_backend: str = "auto"
    tabpfn_version: str = "v3.5"
    tabpfn_train_table: str = "models/windows.csv"
    tabpfn_fit_mode: str = "fit_with_cache"
    tabpfn_estimators: int = 0
    check_every_s: float = 300.0
    max_wakes: int = 3
    snooze_min: float = 60.0
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    temporal_address: str = "localhost:7233"
    task_queue: str = "night-watch"
    sentry_dsn: str = ""

    @classmethod
    def from_env(cls, dotenv: str | None = ".env") -> "Settings":
        if dotenv:
            load_dotenv(dotenv)
        return cls(
            runs_dir=_env("NW_RUNS_DIR", "runs"),
            ollama_url=_env("NW_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            vision_model=_env("NW_VISION_MODEL", "gemma4:e4b"),
            text_model=_env("NW_TEXT_MODEL", "gemma4:e4b"),
            vision_view=_env("NW_VISION_VIEW", "head").lower(),
            vision_think=_flag("NW_VISION_THINK", False),
            vision_max_tokens=int(_env("NW_VISION_MAX_TOKENS", "1500") or 1500),
            vision_snapshots=int(_env("NW_VISION_SNAPSHOTS", "2") or 2),
            vision_reference=_flag("NW_VISION_REFERENCE", False),
            vision_max_side=int(_env("NW_VISION_MAX_SIDE", "1024") or 1024),
            tabpfn_backend=_env("NW_TABPFN_BACKEND", "auto"),
            tabpfn_version=_env("NW_TABPFN_VERSION", "v3.5"),
            tabpfn_train_table=_env("NW_TABPFN_TRAIN_TABLE", "models/windows.csv"),
            tabpfn_fit_mode=_env("NW_TABPFN_FIT_MODE", "fit_with_cache"),
            tabpfn_estimators=int(_env("NW_TABPFN_ESTIMATORS", "0") or 0),
            check_every_s=float(_env("NW_CHECK_EVERY_S", "300") or 300),
            max_wakes=int(_env("NW_MAX_WAKES", "3") or 3),
            snooze_min=float(_env("NW_SNOOZE_MIN", "60") or 60),
            ntfy_topic=_env("NW_NTFY_TOPIC"),
            ntfy_server=_env("NW_NTFY_SERVER", "https://ntfy.sh").rstrip("/"),
            telegram_token=_env("NW_TELEGRAM_TOKEN"),
            telegram_chat_id=_env("NW_TELEGRAM_CHAT_ID"),
            temporal_address=_env("TEMPORAL_ADDRESS", "localhost:7233"),
            task_queue=_env("NW_TASK_QUEUE", "night-watch"),
            sentry_dsn=_env("SENTRY_DSN"),
        )
