"""Tiny Ollama client. Local model, local images, no API key."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import tracing


class OllamaError(RuntimeError):
    pass


class ModelMissing(OllamaError):
    """The model is not pulled. Retrying will not help."""


@dataclass
class ChatResult:
    text: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int


def image_b64(path: str | Path, max_side: int = 1024) -> str:
    """Base64 PNG, downscaled if huge (keeps prompt cost predictable)."""
    p = Path(path)
    try:
        from PIL import Image
        import io

        with Image.open(p) as im:
            im = im.convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        return base64.b64encode(p.read_bytes()).decode("ascii")


class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434", timeout_s: float = 600.0, retries: int = 3):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries

    def chat(self, model: str, prompt: str, images_b64: list[str] | None = None,
             schema: dict | None = None, temperature: float = 0.0, num_ctx: int = 8192,
             purpose: str = "") -> ChatResult:
        message: dict = {"role": "user", "content": prompt}
        if images_b64:
            message["images"] = images_b64
        payload: dict = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [message],
            "options": {"temperature": temperature, "num_ctx": num_ctx, "seed": 0},
        }
        if schema is not None:
            payload["format"] = schema

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            start = time.time()
            with tracing.gen_ai_span(model, "chat", purpose) as span:
                try:
                    r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout_s)
                except requests.RequestException as exc:
                    last_exc = OllamaError(f"cannot reach Ollama at {self.url}: {exc}")
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code == 404 and "not found" in r.text.lower():
                    raise ModelMissing(f"model {model!r} is not pulled. Run: ollama pull {model}")
                if r.status_code == 400 and "think" in r.text.lower() and "think" in payload:
                    payload.pop("think")  # model without a thinking switch
                    continue
                if r.status_code != 200:
                    last_exc = OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
                    time.sleep(2 * (attempt + 1))
                    continue
                data = r.json()
                text = (data.get("message") or {}).get("content", "")
                res = ChatResult(text=text, model=model, latency_s=time.time() - start,
                                 input_tokens=int(data.get("prompt_eval_count") or 0),
                                 output_tokens=int(data.get("eval_count") or 0))
                span.set_data("gen_ai.usage.input_tokens", res.input_tokens)
                span.set_data("gen_ai.usage.output_tokens", res.output_tokens)
                span.set_data("gen_ai.response.model", model)
                span.set_data("nightwatch.latency_s", round(res.latency_s, 3))
                return res
        raise last_exc or OllamaError("Ollama call failed")

    def chat_json(self, model: str, prompt: str, schema: dict, images_b64: list[str] | None = None,
                  purpose: str = "") -> tuple[dict | None, ChatResult]:
        res = self.chat(model, prompt, images_b64=images_b64, schema=schema, purpose=purpose)
        try:
            return json.loads(res.text), res
        except json.JSONDecodeError:
            # Some models wrap JSON in prose; take the outermost braces.
            s, e = res.text.find("{"), res.text.rfind("}")
            if s != -1 and e > s:
                try:
                    return json.loads(res.text[s:e + 1]), res
                except json.JSONDecodeError:
                    pass
            return None, res
