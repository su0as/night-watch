"""Getting a message to a sleeping human: console always, plus ntfy and/or Telegram."""

from __future__ import annotations

import time
from pathlib import Path

from .config import Settings


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


class Notifier:
    def __init__(self, settings: Settings, quiet: bool = False):
        self.s = settings
        self.quiet = quiet
        self.sent: list[dict] = []

    def send(self, title: str, message: str, urgent: bool = False, image: str | Path | None = None) -> list[str]:
        """Returns the channels that accepted the message."""
        channels: list[str] = []
        stamp = time.strftime("%H:%M:%S")
        if not self.quiet:
            mark = "WAKE" if urgent else "info"
            print(f"[{stamp}] [{mark}] {title}: {message}" + (f" ({image})" if image else ""), flush=True)
        channels.append("console")
        img = Path(image) if image else None
        if img is not None and not img.is_file():
            img = None
        if self.s.ntfy_topic:
            try:
                self._ntfy(title, message, urgent, img)
                channels.append("ntfy")
            except Exception as exc:  # noqa: BLE001
                print(f"[notify] ntfy failed: {exc}", flush=True)
        if self.s.telegram_token and self.s.telegram_chat_id:
            try:
                self._telegram(title, message, img)
                channels.append("telegram")
            except Exception as exc:  # noqa: BLE001
                print(f"[notify] telegram failed: {exc}", flush=True)
        self.sent.append({"t": time.time(), "title": title, "message": message, "urgent": urgent,
                          "image": str(img) if img else None, "channels": channels})
        return channels

    def _ntfy(self, title: str, message: str, urgent: bool, image: Path | None) -> None:
        import requests

        url = f"{self.s.ntfy_server}/{self.s.ntfy_topic}"
        headers = {
            "Title": _ascii(title),
            "Priority": "5" if urgent else "3",
            "Tags": "rotating_light,robot" if urgent else "sleeping,robot",
        }
        if image is not None:
            headers["Filename"] = image.name
            headers["Message"] = _ascii(message)[:3000]
            with open(image, "rb") as f:
                r = requests.put(url, data=f, headers=headers, timeout=30)
        else:
            r = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=30)
        r.raise_for_status()

    def _telegram(self, title: str, message: str, image: Path | None) -> None:
        import requests

        base = f"https://api.telegram.org/bot{self.s.telegram_token}"
        text = f"{title}\n{message}"
        if image is not None:
            with open(image, "rb") as f:
                r = requests.post(f"{base}/sendPhoto", data={"chat_id": self.s.telegram_chat_id,
                                                            "caption": text[:1000]},
                                  files={"photo": (image.name, f)}, timeout=60)
        else:
            r = requests.post(f"{base}/sendMessage", data={"chat_id": self.s.telegram_chat_id,
                                                          "text": text[:4000]}, timeout=30)
        r.raise_for_status()
