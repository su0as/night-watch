"""Optional Sentry agent tracing. If SENTRY_DSN is unset or sentry-sdk is missing, every
function here is a no-op, so the rest of the code never has to care."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

_enabled = False


def init(dsn: str | None = None, release: str | None = None) -> bool:
    global _enabled
    dsn = dsn or os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(dsn=dsn, traces_sample_rate=1.0, send_default_pii=False,
                        release=release or "night-watch@0.1.0")
        _enabled = True
    except Exception:  # noqa: BLE001
        _enabled = False
    return _enabled


class _NullSpan:
    def set_data(self, *_a: Any, **_k: Any) -> None:
        pass

    def set_tag(self, *_a: Any, **_k: Any) -> None:
        pass


@contextlib.contextmanager
def transaction(name: str, op: str = "nightwatch.check") -> Iterator[Any]:
    if not _enabled:
        yield _NullSpan()
        return
    import sentry_sdk

    with sentry_sdk.start_transaction(op=op, name=name) as tx:
        yield tx


@contextlib.contextmanager
def gen_ai_span(model: str, operation: str = "chat", purpose: str = "") -> Iterator[Any]:
    """A gen_ai span in the shape Sentry's AI agent monitoring expects."""
    if not _enabled:
        yield _NullSpan()
        return
    import sentry_sdk

    with sentry_sdk.start_span(op=f"gen_ai.{operation}", name=f"{operation} {model}") as span:
        span.set_data("gen_ai.system", "ollama")
        span.set_data("gen_ai.operation.name", operation)
        span.set_data("gen_ai.request.model", model)
        if purpose:
            span.set_data("nightwatch.purpose", purpose)
        yield span


def capture_message(message: str, level: str = "warning", **tags: str) -> None:
    if not _enabled:
        return
    import sentry_sdk

    with sentry_sdk.new_scope() as scope:
        for k, v in tags.items():
            scope.set_tag(k, v)
        sentry_sdk.capture_message(message, level=level)
