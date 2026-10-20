"""The morning report: what happened while you slept, in one page.

Numbers come from code. Gemma is only allowed to *phrase* them, and its paragraph is
checked: if it contains any number that is not in the facts, it is thrown away and a
plain template is used instead. The report says which one you got.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict
from pathlib import Path

from .features import KEYS
from .policy import Alert, PolicyState
from .runlog import RunReader, write_json_atomic


# ----------------------------------------------------------------------------- facts

def _fmt_clock(t: float | None) -> str:
    return time.strftime("%H:%M", time.localtime(t)) if t else "?"


def night_facts(run_dir: str | Path, state: PolicyState, watch_started: float | None = None,
                watch_ended: float | None = None) -> dict:
    reader = RunReader(run_dir)
    m = reader.manifest()
    rows = reader.metrics()
    frames = reader.frames()
    rews = [r.get(KEYS["rew"]) for r in rows if isinstance(r.get(KEYS["rew"]), (int, float))]
    rews = [float(v) for v in rews if math.isfinite(float(v))]
    first_step = int(rows[0]["step"]) if rows else 0
    last_step = int(rows[-1]["step"]) if rows else 0
    started = watch_started or (rows[0]["t"] if rows else None)
    ended = watch_ended or time.time()
    return {
        "run": reader.run_id,
        "env": m.get("env_id", "?"),
        "status": m.get("status", "running"),
        "steps_start": first_step,
        "steps_end": last_step,
        "steps_done": max(0, last_step - first_step),
        "reward_start": round(rews[0]) if rews else None,
        "reward_end": round(rews[-1]) if rews else None,
        "reward_best": round(max(rews)) if rews else None,
        "test_runs": len(frames),
        "wakes": len(state.wakes),
        "notes": len(state.notes),
        "suppressed": state.suppressed,
        "watch_from": _fmt_clock(started),
        "watch_to": _fmt_clock(ended),
        "hours": round((ended - started) / 3600, 1) if started else None,
        "first_problem": state.wakes[0].message if state.wakes else (state.notes[0].message if state.notes else None),
        "first_problem_at": _fmt_clock(state.wakes[0].t) if state.wakes else None,
    }


def template_summary(f: dict) -> str:
    parts = [f"Watched {f['run']} from {f['watch_from']} to {f['watch_to']}."]
    if f["reward_start"] is not None:
        parts.append(f"Training went from step {f['steps_start']:,} to {f['steps_end']:,}; "
                     f"reward moved from {f['reward_start']} to {f['reward_end']} (best {f['reward_best']}).")
    if f["wakes"]:
        parts.append(f"I woke you {f['wakes']} time(s); the first was at {f['first_problem_at']}: {f['first_problem']}")
    elif f["notes"]:
        parts.append(f"Nothing worth waking you for. {f['notes']} thing(s) to look at below.")
    else:
        parts.append("Quiet night. Nothing looked wrong.")
    return " ".join(parts)


# ----------------------------------------------------------------------------- number guard

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def allowed_numbers(facts: dict) -> set[str]:
    allowed: set[str] = set()
    for v in facts.values():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            allowed.add(str(v))
            allowed.add(str(int(v)) if float(v).is_integer() else str(v))
            if isinstance(v, float):
                allowed.add(f"{v:.1f}")
        else:
            allowed |= _numbers_in(str(v))
    return allowed


def numbers_ok(text: str, facts: dict) -> tuple[bool, set[str]]:
    extra = _numbers_in(text) - allowed_numbers(facts)
    return (not extra), extra


def llm_summary(facts: dict, client, model: str) -> tuple[str, str]:
    """(text, source) where source is 'gemma' or 'template (reason)'."""
    if client is None:
        return template_summary(facts), "template"
    prompt = (
        "Write a short, calm morning note (3 sentences max) to a robotics student about their overnight "
        "training run. Use ONLY the facts below. Do not invent numbers, causes or advice. "
        "Plain words, no headings, no lists.\n\nFacts:\n"
        + "\n".join(f"- {k}: {v}" for k, v in facts.items() if v is not None)
    )
    try:
        res = client.chat(model, prompt, purpose="morning_summary")
    except Exception as exc:  # noqa: BLE001
        return template_summary(facts), f"template (model unavailable: {type(exc).__name__})"
    text = res.text.strip()
    ok, extra = numbers_ok(text, facts)
    if not ok or not text:
        return template_summary(facts), f"template (model used numbers not in the facts: {sorted(extra)[:5]})"
    return text, "gemma"


# ----------------------------------------------------------------------------- pictures

def timelapse_gif(run_dir: str | Path, out: Path, max_frames: int = 60, width: int = 480) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    reader = RunReader(run_dir)
    frames = [f for f in reader.frames() if f.get("frames") or f.get("sheet")]
    if not frames:
        return None
    if len(frames) > max_frames:
        stride = math.ceil(len(frames) / max_frames)
        frames = frames[::stride] + ([frames[-1]] if frames[-1] not in frames[::stride] else [])
    imgs = []
    for f in frames:
        rel = (f.get("frames") or [None])[0] or f.get("sheet")
        p = reader.resolve(rel)
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB")
        if im.width != width:
            im = im.resize((width, int(im.height * width / im.width)))
        d = ImageDraw.Draw(im)
        label = f"step {int(f['step']):,}   {_fmt_clock(f.get('t'))}"
        d.rectangle([0, 0, width, 22], fill=(0, 0, 0))
        d.text((8, 5), label, fill=(255, 255, 255))
        imgs.append(im)
    if not imgs:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=350, loop=0, optimize=True)
    return out


def reward_chart(run_dir: str | Path, out: Path, alerts: list[Alert]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = RunReader(run_dir).metrics()
    pts = [(r["step"], r[KEYS["rew"]]) for r in rows
           if isinstance(r.get(KEYS["rew"]), (int, float)) and math.isfinite(r[KEYS["rew"]])]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#2b6cb0", lw=1.6)
    for a in alerts:
        ax.axvline(a.step, color="#c53030" if a.level == "wake" else "#d69e2e", lw=1, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("episode reward")
    ax.grid(alpha=0.25)
    ax.set_title("Reward overnight (red = woke you, amber = morning note)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- report

def write_morning_report(run_dir: str | Path, state: PolicyState, client=None, text_model: str = "gemma4:e4b",
                         watch_started: float | None = None, watch_ended: float | None = None,
                         profile: str = "nightwatch") -> dict:
    run_dir = Path(run_dir)
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = night_facts(run_dir, state, watch_started, watch_ended)
    summary, source = llm_summary(facts, client, text_model)
    gif = timelapse_gif(run_dir, out_dir / "timelapse.gif")
    chart = reward_chart(run_dir, out_dir / "reward.png", state.wakes + state.notes)

    lines = [f"# Night Watch: {facts['run']}", "", summary, "", f"_Summary written by: {source}_", ""]
    if gif:
        lines += ["![time-lapse of the night](timelapse.gif)", ""]
    if chart:
        lines += ["![reward overnight](reward.png)", ""]
    lines += ["| | |", "|---|---|"]
    for k in ("env", "status", "steps_start", "steps_end", "reward_start", "reward_end", "reward_best",
              "test_runs", "wakes", "notes", "suppressed", "watch_from", "watch_to"):
        lines.append(f"| {k.replace('_', ' ')} | {facts.get(k)} |")
    lines.append("")
    if state.wakes:
        lines += ["## Woke you", ""]
        for a in state.wakes:
            lines.append(f"- **{_fmt_clock(a.t)}** step {a.step:,} [{a.source}/{a.kind}] {a.message}")
            if a.evidence.get("sheet"):
                lines.append(f"  ![{a.kind}](../{a.evidence['sheet']})")
        lines.append("")
    if state.notes:
        lines += ["## For the morning", ""]
        for a in state.notes:
            lines.append(f"- {_fmt_clock(a.t)} step {a.step:,} [{a.source}/{a.kind}] {a.message}")
        lines.append("")
    lines.append(f"_Watcher profile: {profile}. Generated {time.strftime('%Y-%m-%d %H:%M')}._")
    md = out_dir / "morning.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    write_json_atomic(out_dir / "night.json", {
        "facts": facts, "summary": summary, "summary_source": source, "profile": profile,
        "wakes": [asdict(a) for a in state.wakes], "notes": [asdict(a) for a in state.notes],
    })
    return {"markdown": str(md), "gif": str(gif) if gif else None, "chart": str(chart) if chart else None,
            "summary": summary, "summary_source": source, "facts": facts}
