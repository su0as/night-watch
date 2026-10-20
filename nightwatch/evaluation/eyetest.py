"""python -m nightwatch eyetest: can Gemma actually tell which way up the robot is?

A controlled check of the vision detector, separate from training. The robot is dropped
into a known pose (right way up, on its back, on its side), the physics settles it with
small random motor twitches, and four snapshots become a contact sheet, exactly like the
sheets the trainer writes. Ground truth is the torso angle from the simulator.

The same VisionJudge that runs at night then looks at every sheet, once per mode
(e.g. answer straight away vs think first), and the results go to docs/EYETEST.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from ..runlog import append_jsonl, read_jsonl, write_json_atomic

CLASSES = {"upright": 0.0, "upside_down": 3.14159, "on_its_side": 1.5708}


def truth_class(tilt_deg: float) -> str:
    if tilt_deg < 60.0:
        return "upright"
    if tilt_deg > 120.0:
        return "upside_down"
    return "on_its_side"


@dataclass
class Sheet:
    sheet: str
    intended: str
    tilt_deg: float
    truth: str
    seed: int
    mirror: bool
    frames: list | None = None
    size: list | None = None
    heads: list | None = None


def make_sheets(out_dir: str | Path, env_id: str = "HalfCheetah-v5", per_class: int = 6,
                mirror: bool = False, width: int = 1024, height: int = 768,
                classes: tuple[str, ...] | None = None) -> list[Sheet]:
    """Render controlled contact sheets plus a reference sheet (seed 0, untouched start pose).

    A planar robot (HalfCheetah, Hopper, Walker2d) cannot rest on its side: it falls one way or the
    other, so only upright and upside down are tested for those."""
    from ..truth import env_family

    if classes is None:
        classes = ("upright", "upside_down") if env_family(env_id) == "planar" else tuple(CLASSES)
    import gymnasium as gym
    import numpy as np
    from PIL import Image

    from ..cameras import RobotCameras
    from ..train import contact_sheet
    from ..truth import pose_from_state

    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id)
    env.reset(seed=0)
    u = env.unwrapped
    cams = RobotCameras(env_id, width=width, height=height)
    if not cams.attach(env, head=True, matte=not mirror):
        raise RuntimeError(f"cannot render: {cams.error}")

    def snap(rng, n=4, steps_between=8, twitch=0.3) -> tuple[list, list]:
        frames, heads = [], []
        for _ in range(n):
            for _ in range(steps_between):
                env.step(rng.uniform(-twitch, twitch, size=env.action_space.shape))
            frames.append(cams.side(u.data.qpos, u.data.qvel))
            h = cams.head(u.data.qpos, u.data.qvel)
            if h is not None:
                heads.append(h)
        return frames, heads

    # Reference: what the trainer's first sheet looks like (untrained robot, start pose).
    env.reset(seed=0)
    ref = contact_sheet(snap(np.random.default_rng(0), twitch=0.05)[0])
    ref.save(out / "frames" / "reference_sheet.png")

    sheets: list[Sheet] = []
    for cls in classes:
        pitch = CLASSES[cls]
        for k in range(per_class):
            seed = 1000 * (list(CLASSES).index(cls) + 1) + k
            rng = np.random.default_rng(seed)
            env.reset(seed=seed)
            qpos = u.init_qpos.copy()
            qvel = np.zeros_like(u.init_qvel)
            qpos[2] = pitch + rng.uniform(-0.25, 0.25)
            qpos[3:] = rng.uniform(-0.5, 0.5, size=qpos[3:].shape)
            if cls != "upright":
                qpos[1] = -0.3 if cls == "upside_down" else -0.2
            u.set_state(qpos, qvel)
            for _ in range(100):  # let it fall and settle
                env.step(rng.uniform(-0.3, 0.3, size=env.action_space.shape))
            frames, heads = snap(rng)
            pose = pose_from_state(env_id, u.data.qpos, u.data.qvel)
            rel = f"frames/{cls}_{k}_sheet.png"
            contact_sheet(frames).save(out / rel)
            rels, hrels = [], []
            for j, f in enumerate(frames):
                fr = f"frames/{cls}_{k}_{j}.png"
                Image.fromarray(f).save(out / fr)
                rels.append(fr)
            for j, h in enumerate(heads):
                hr = f"frames/{cls}_{k}_head{j}.png"
                Image.fromarray(h).save(out / hr)
                hrels.append(hr)
            sheets.append(Sheet(rel, cls, round(pose.tilt_deg, 1), truth_class(pose.tilt_deg), seed, mirror,
                                rels, [width, height], hrels))
    cams.close()
    env.close()
    write_json_atomic(out / "sheets.json", {"env_id": env_id, "mirror": mirror, "size": [width, height],
                                            "sheets": [s.__dict__ for s in sheets]})
    return sheets



def config_name(size: tuple[int, int], mirror: bool) -> str:
    return f"{size[0]}x{size[1]}{'-mirror' if mirror else ''}"


def run_eyetest(root: str | Path, settings, modes: list[str], env_id: str = "HalfCheetah-v5",
                per_class: int = 6, mirror: bool = False, out_md: str | Path | None = None,
                size: tuple[int, int] = (1024, 768), sheet_only: bool = False) -> dict:
    """modes: 'head' (head camera, the default watcher), 'side' (side camera, answer straight away),
    'side-think' (side camera, reason first; very slow on a laptop).
    sheet_only=True shows the side judge the 2x2 contact sheet instead of full-size snapshots (the old way).

    Every look is appended to <root>/eyetest.jsonl, and the markdown is rebuilt from that whole log,
    so running a few configurations one after another gives one comparison table."""
    from ..detectors.vision import make_judge
    from ..llm import OllamaClient

    root = Path(root)
    out = root / config_name(size, mirror)
    meta = out / "sheets.json"
    old = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    if old.get("env_id") == env_id and len(old.get("sheets", [])) >= per_class * 2:
        sheets = [Sheet(**s) for s in old["sheets"]]
    else:
        sheets = make_sheets(out, env_id, per_class=per_class, mirror=mirror, width=size[0], height=size[1])
    client = OllamaClient(settings.ollama_url)
    results: dict[str, dict] = {}
    log = root / "eyetest.jsonl"
    for mode in modes:
        head = mode == "head"
        judge = make_judge(settings, client, think=(mode == "side-think"), view="head" if head else "side",
                           cache_name=f"eyetest_cache_{mode}.jsonl",
                           **({"snapshots": 0} if sheet_only and not head else {}))
        if head:
            shown, version = f"head camera, {judge.head_snapshots} snapshots", judge.head_version
        else:
            shown = "side sheet" if sheet_only else f"side camera, {judge.snapshots} snapshot(s)"
            version = judge.version
        print(f"[eyetest] {mode}: {settings.vision_model} {version}, {config_name(size, mirror)}, {shown}",
              flush=True)
        rows = []
        for s in sheets:
            row_in = {"step": s.seed, "sheet": s.sheet, "frames": [] if sheet_only else (s.frames or []),
                      "head": (s.heads or []) if head else []}
            v = judge.judge(out, row_in, "frames/reference_sheet.png", env_id)
            # Right wake decision: wake for upside down, stay quiet otherwise.
            correct = v.ok and (v.wrong == (s.truth == "upside_down"))
            row = {"mode": mode, "config": config_name(size, mirror), "shown": shown, "sheet": s.sheet,
                   "truth": s.truth, "tilt_deg": s.tilt_deg, "posture": v.posture, "motion": v.motion,
                   "feet": v.feet, "wake_worthy": v.wrong, "correct_wake": correct,
                   "exact": v.ok and v.posture == s.truth, "latency_s": v.latency_s,
                   "in_tokens": v.input_tokens, "out_tokens": v.output_tokens, "ok": v.ok, "why": v.why,
                   "error": v.error[:200], "model": settings.vision_model, "prompt_version": version}
            rows.append(row)
            append_jsonl(log, row | {"t": time.time()})
            print(f"[eyetest] {mode:5s} {s.sheet:32s} truth={s.truth:12s} gemma={v.posture:12s} "
                  f"feet={v.feet:10s} {'OK ' if correct else 'BAD'} {v.latency_s:5.1f}s"
                  + (f" ERROR {v.error[:100]}" if not v.ok else ""), flush=True)
        results[mode] = summarise_mode(rows) | {"version": version, "rows": rows}
    if out_md:
        write_markdown(out_md, read_jsonl(log), env_id)
    return results


def summarise_mode(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    by_truth: dict[str, list[dict]] = {}
    for r in ok:
        by_truth.setdefault(r["truth"], []).append(r)
    lat = [r["latency_s"] for r in ok if r["latency_s"]]
    return {
        "sheets": len(rows), "answered": len(ok),
        "wake_decision_correct": sum(r["correct_wake"] for r in ok),
        "posture_exact": sum(r["exact"] for r in ok),
        "per_truth": {k: {"n": len(v), "wake_correct": sum(r["correct_wake"] for r in v),
                          "said": _counts([r["posture"] for r in v])} for k, v in by_truth.items()},
        "median_latency_s": round(median(lat), 1) if lat else None,
        "median_out_tokens": int(median([r["out_tokens"] for r in ok])) if ok else None,
        "median_in_tokens": int(median([r.get("in_tokens", 0) for r in ok])) if ok else None,
    }


def _counts(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def write_markdown(path: str | Path, log_rows: list[dict], env_id: str) -> None:
    """One table row per (configuration, what Gemma was shown, prompt version, mode); latest run of each."""
    groups: dict[tuple, list[dict]] = {}
    for r in log_rows:
        if "config" not in r:
            continue
        key = (r["model"], r["config"], r["shown"], r["prompt_version"], r["mode"])
        groups.setdefault(key, []).append(r)
    # keep only the most recent pass over each group's sheets
    latest: dict[tuple, list[dict]] = {}
    for key, rows in groups.items():
        by_sheet: dict[str, dict] = {}
        for r in sorted(rows, key=lambda x: x.get("t", 0)):
            by_sheet[r["sheet"]] = r
        latest[key] = list(by_sheet.values())
    L = ["# Gemma eyesight test", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m nightwatch eyetest` (robot `{env_id}`).", "",
         "The robot is dropped into a known pose (the right way up, or on its back), the physics settles it "
         "with small random motor twitches, and four snapshots are taken, exactly like a test episode during "
         "training. The truth is the torso angle from the simulator (upright < 60 degrees, upside down > 120). "
         "The judge is the same code that watches at night; only what it is shown changes.", "",
         "| model | snapshots | floor | Gemma is shown | mode | right wake decision | upright left alone "
         "| upside down caught | image+prompt tokens | seconds per look |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for (model, config, shown, version, mode), rows in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][4])):
        s = summarise_mode(rows)
        pt = s["per_truth"]
        up = pt.get("upright", {"n": 0, "wake_correct": 0})
        dn = pt.get("upside_down", {"n": 0, "wake_correct": 0})
        size, _, floor = config.partition("-")
        L.append(f"| `{model}` | {size} | {'mirror' if floor == 'mirror' else 'matte'} | {shown} | {mode} "
                 f"| **{s['wake_decision_correct']} / {s['answered']}** | {up['wake_correct']} / {up['n']} "
                 f"| {dn['wake_correct']} / {dn['n']} | {s['median_in_tokens']} | {s['median_latency_s']} |")
    L += ["", "What Gemma called each pose:", ""]
    for (model, config, shown, version, mode), rows in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][4])):
        s = summarise_mode(rows)
        parts = []
        for truth, d in s["per_truth"].items():
            said = ", ".join(f"{k} x{v}" for k, v in sorted(d["said"].items(), key=lambda kv: -kv[1]))
            parts.append(f"truly {truth}: {said}")
        L.append(f"- {config}, {shown}, {mode} (`{version}`): " + "; ".join(parts))
    L += ["", "Ollama gives Gemma 4 roughly one image token per 48x48 pixels, so a 2x2 sheet of 320x240 "
          "snapshots is ~130 image tokens for all four snapshots, while one 1024x768 snapshot is ~336. "
          "Seconds per look were measured on an Intel i7-1165G7 laptop with Gemma on the Iris Xe iGPU (Vulkan)."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
