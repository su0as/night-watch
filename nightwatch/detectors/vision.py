"""Gemma looks at the robot.

Every few thousand training steps the trainer runs a short test episode and saves four
snapshots from a side camera (for humans: reports, alerts) and four from a head camera bolted
to the robot (for Gemma, see cameras.py). For each head-camera snapshot Gemma answers one
question in a fixed JSON schema: where is the floor in this picture? Plain code then turns the
four answers into a posture and motion (cameras.posture_from_floor). Only those enum fields
decide a wake-up, never free text, so a chatty model cannot talk itself into waking you.

The side-view judge below is kept for runs recorded without a head camera, and for the
eyetest, because it is the honest baseline: Gemma 4 E4B could not tell upright from upside
down from the side view (it gave the same answer for both).

Why separate full-size snapshots and not the contact sheet: Ollama gives Gemma 4 about one
image token per 48x48 pixels. A 640x480 sheet of four snapshots is ~130 tokens, ~33 per
snapshot, and at that size Gemma 4 E4B gave the same posture for every picture, upright or
upside down, whatever the prompt said. One 1024x768 snapshot is ~336 tokens, and the same
model got them right. `python -m nightwatch eyetest` measures this on controlled poses.

The reference image (the run's very first sheet: the robot the right way up) is optional
(NW_VISION_REFERENCE=1); it did not help in the eyetest and doubles the image tokens.
Thinking (NW_VISION_THINK=1) is optional too: ~10x the output tokens.

Verdicts are cached per sheet in vision_cache.jsonl, so replays and re-checks are free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..llm import OllamaClient, OllamaError, image_b64
from ..runlog import append_jsonl, read_jsonl
from .base import Observation, Signal

PROMPT_VERSION = "v3"
HEAD_PROMPT_VERSION = "head-v3"

HEAD_PROMPT = ("This picture comes from a camera fixed inside a robot's body, looking straight ahead. "
               "Answer two questions about the picture.\n"
               "1. Where is the checkered floor: in the bottom half, in the top half, filling the whole picture, "
               "along the left or right side, or not visible at all?\n"
               "2. Where is the plain grey sky: in the top half, in the bottom half, filling the whole picture, "
               "along the left or right side, or not visible at all?")

_POS = ["bottom_half", "top_half", "whole_picture", "left_or_right", "not_visible"]
HEAD_SCHEMA = {
    "type": "object",
    "properties": {"floor": {"type": "string", "enum": _POS}, "sky": {"type": "string", "enum": _POS}},
    "required": ["floor", "sky"],
}

POSTURES = ["upright", "upside_down", "on_its_side", "mid_flip", "broken", "unclear"]
MOTIONS = ["moving_forward", "moving_backward", "spinning", "stuck", "jittering", "unclear"]
# Wake-worthy. Matches the ground-truth definition (tilt > 120 degrees = flipped).
WRONG_POSTURES = {"upside_down", "broken"}
WRONG_MOTIONS = {"spinning"}
# Worth a morning note only: a running cheetah pitches a lot, so "on its side" is ambiguous.
ODD_POSTURES = {"on_its_side"}

SCHEMA = {
    "type": "object",
    "properties": {
        "robot_visible": {"type": "boolean"},
        "feet": {"type": "string", "enum": ["below_body", "above_body", "mixed", "unclear"]},
        "posture": {"type": "string", "enum": POSTURES},
        "motion": {"type": "string", "enum": MOTIONS},
        "why": {"type": "string"},
    },
    "required": ["robot_visible", "feet", "posture", "motion", "why"],
}

ROBOTS = {
    "HalfCheetah": ("a two-legged cheetah robot seen from the side. Its body is a long orange horizontal "
                    "bar with the head at one end; its two legs hang below the body and the pink lower legs "
                    "and feet touch the floor",
                    "run forward along the floor with its back up and its legs underneath it"),
    "Hopper": ("a one-legged hopping robot seen from the side: a vertical torso on a single leg",
               "hop forward while staying upright on its foot"),
    "Walker2d": ("a two-legged walking robot seen from the side: a vertical torso on two legs",
                 "walk forward upright on its feet"),
    "Ant": ("a four-legged ant robot: a round body with four legs spreading out and down to the floor",
            "walk forward with its body up and its legs underneath"),
    "Humanoid": ("a humanoid robot with a torso, two arms and two legs", "walk forward upright"),
}

PROMPT = """You are watching a robot learn in a physics simulator overnight while its owner sleeps.

The robot is {robot}. It is supposed to {task}.

{images}

In each snapshot, find the feet: are they below the body and near the floor, or above the body,
pointing up into the air? Is the body up high with the legs under it, or down on the floor?

Then answer:
- feet: below_body, above_body, mixed (it changes between snapshots), or unclear
- posture: upright (the right way up), upside_down (back on the floor, legs in the air),
  on_its_side, mid_flip, broken (body parts in impossible places, flying apart), or unclear
- motion across the snapshots: moving_forward, moving_backward, spinning (rotating/somersaulting),
  stuck, jittering, or unclear
- why: at most 15 words, about what you saw.
"""


def images_text(n_snapshots: int, reference: bool, sheet: bool) -> str:
    what = ("one image with four snapshots (left to right, top to bottom)" if sheet
            else f"{n_snapshots} snapshot{'s' if n_snapshots != 1 else ''}, in order,")
    if reference:
        return ("Image 1 is a REFERENCE taken at the very start of training: the robot the right way up.\n"
                f"After it comes {what} from the robot's latest test run.")
    return f"You get {what} from the robot's latest test run."


def pick_snapshots(frame_row: dict, n: int) -> tuple[list[str], bool]:
    """The last `n` full-size snapshots of a test episode, or the contact sheet for old runs."""
    frames = [f for f in (frame_row.get("frames") or []) if f]
    if frames and n > 0:
        return frames[-n:], False
    return [frame_row["sheet"]], True


def robot_text(env_id: str) -> tuple[str, str]:
    for key, val in ROBOTS.items():
        if env_id.startswith(key):
            return val
    return ("a simulated robot", "move forward the right way up")


@dataclass
class Verdict:
    step: int
    sheet: str
    model: str
    prompt_version: str
    ok: bool
    robot_visible: bool = True
    posture: str = "unclear"
    motion: str = "unclear"
    looks_wrong: bool = False
    feet: str = "unclear"
    think: bool = False
    why: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def wrong(self) -> bool:
        return self.ok and self.robot_visible and (self.posture in WRONG_POSTURES or self.motion in WRONG_MOTIONS)

    @property
    def odd(self) -> bool:
        return self.ok and self.robot_visible and self.posture in ODD_POSTURES

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)


class VisionJudge:
    """Asks Gemma about one contact sheet, with an on-disk cache."""

    def __init__(self, client: OllamaClient | None, model: str = "gemma4:e4b", think: bool = False,
                 max_tokens: int = 1500, cache_name: str = "vision_cache.jsonl", snapshots: int = 2,
                 reference: bool = False, max_side: int = 1024, view: str = "head", head_snapshots: int = 4):
        self.client = client
        self.model = model
        self.think = think
        self.max_tokens = max_tokens
        self.cache_name = cache_name
        self.snapshots = snapshots
        self.reference = reference
        self.max_side = max_side
        self.view = view
        self.head_snapshots = head_snapshots

    @property
    def version(self) -> str:
        """Everything that changes what Gemma sees or is asked, so the cache never mixes them."""
        return f"{PROMPT_VERSION}-n{self.snapshots}{'-ref' if self.reference else ''}-px{self.max_side}"

    @property
    def head_version(self) -> str:
        return f"{HEAD_PROMPT_VERSION}-n{self.head_snapshots}"

    def _uses_head(self, frame_row: dict) -> bool:
        return self.view == "head" and bool([h for h in (frame_row.get("head") or []) if h])

    def cached(self, run_dir: str | Path, sheet: str, head: bool | None = None) -> Verdict | None:
        versions = {True: [self.head_version], False: [self.version], None: [self.head_version, self.version]}[head]
        if self.view != "head":
            versions = [v for v in versions if v != self.head_version] or [self.version]
        rows = read_jsonl(Path(run_dir) / self.cache_name)
        for version in versions:
            think = False if version == self.head_version else bool(self.think)
            for row in reversed(rows):
                key = (row.get("sheet"), row.get("model"), row.get("prompt_version"), bool(row.get("think", False)))
                if key == (sheet, self.model, version, think):
                    return Verdict.from_dict(row)
        return None

    def judge(self, run_dir: str | Path, frame_row: dict, reference_sheet: str, env_id: str) -> Verdict:
        sheet = frame_row["sheet"]
        head = self._uses_head(frame_row)
        hit = self.cached(run_dir, sheet, head=head)
        if hit is not None:
            return hit
        if head:
            return self._judge_head(Path(run_dir), frame_row)
        robot, task = robot_text(env_id)
        snaps, is_sheet = pick_snapshots(frame_row, self.snapshots)
        use_ref = self.reference and bool(reference_sheet)
        prompt = PROMPT.format(robot=robot, task=task, images=images_text(len(snaps), use_ref, is_sheet))
        root = Path(run_dir)
        if self.client is None:
            return Verdict(step=int(frame_row["step"]), sheet=sheet, model=self.model,
                           prompt_version=self.version, ok=False, think=self.think, error="no model client")
        paths = ([root / reference_sheet] if use_ref else []) + [root / s for s in snaps]
        try:
            data, res = self.client.chat_json(
                self.model, prompt, SCHEMA,
                images_b64=[image_b64(p, max_side=self.max_side) for p in paths],
                purpose="look_at_robot", think=self.think, max_tokens=self.max_tokens)
        except OllamaError as exc:
            # Not cached: a transient failure should be retried next time.
            return Verdict(step=int(frame_row["step"]), sheet=sheet, model=self.model,
                           prompt_version=self.version, ok=False, think=self.think, error=str(exc))
        v = Verdict(step=int(frame_row["step"]), sheet=sheet, model=self.model,
                    prompt_version=self.version, ok=data is not None, think=self.think,
                    latency_s=round(res.latency_s, 2), input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens)
        if data is not None:
            v.robot_visible = bool(data.get("robot_visible", True))
            v.posture = data.get("posture") if data.get("posture") in POSTURES else "unclear"
            v.motion = data.get("motion") if data.get("motion") in MOTIONS else "unclear"
            v.feet = data.get("feet") if data.get("feet") in ("below_body", "above_body", "mixed") else "unclear"
            v.looks_wrong = v.posture in WRONG_POSTURES or v.motion in WRONG_MOTIONS
            v.why = str(data.get("why", ""))[:200]
        else:
            # Truncated by the token cap or not JSON: not cached, so the next check tries again.
            v.error = "unparseable: " + res.text[:200]
            return v
        append_jsonl(root / self.cache_name, asdict(v))
        return v

    def _judge_head(self, root: Path, frame_row: dict) -> Verdict:
        """One short question per head-camera snapshot; the decision is made by code."""
        from ..cameras import posture_from_votes, snapshot_vote

        sheet = frame_row["sheet"]
        heads = [h for h in (frame_row.get("head") or []) if h][-self.head_snapshots:]
        base = dict(step=int(frame_row["step"]), sheet=sheet, model=self.model, prompt_version=self.head_version,
                    think=False)
        if self.client is None:
            return Verdict(**base, ok=False, error="no model client")
        floors, skies, lat, tin, tout = [], [], 0.0, 0, 0
        for rel in heads:
            try:
                data, res = self.client.chat_json(self.model, HEAD_PROMPT, HEAD_SCHEMA,
                                                  images_b64=[image_b64(root / rel, max_side=self.max_side)],
                                                  purpose="look_at_floor", think=False, max_tokens=64)
            except OllamaError as exc:
                return Verdict(**base, ok=False, error=str(exc))  # not cached: retried next check
            lat += res.latency_s
            tin += res.input_tokens
            tout += res.output_tokens
            floor, sky = (data or {}).get("floor"), (data or {}).get("sky")
            floors.append(floor if floor in _POS else "not_visible")
            skies.append(sky if sky in _POS else "not_visible")
        votes = [snapshot_vote(f, k) for f, k in zip(floors, skies)]
        posture, motion, feet = posture_from_votes(votes)
        v = Verdict(**base, ok=True, posture=posture, motion=motion, feet=feet,
                    robot_visible=any(f != "not_visible" for f in floors + skies),
                    why="head camera, floor/sky: " + ", ".join(
                        f"{f.replace('_half', '')}/{k.replace('_half', '')}" for f, k in zip(floors, skies)),
                    latency_s=round(lat, 2), input_tokens=tin, output_tokens=tout)
        v.looks_wrong = v.wrong
        append_jsonl(root / self.cache_name, asdict(v))
        return v


def make_judge(settings, client: OllamaClient | None, **overrides) -> VisionJudge:
    """The one place a VisionJudge is built from Settings, so the live watch, `look`, `eval`
    and `eyetest` all ask Gemma exactly the same way (and share one cache)."""
    kw = dict(model=settings.vision_model, think=settings.vision_think, max_tokens=settings.vision_max_tokens,
              snapshots=settings.vision_snapshots, reference=settings.vision_reference,
              max_side=settings.vision_max_side, view=settings.vision_view)
    kw.update(overrides)
    return VisionJudge(client, **kw)


class VisionDetector:
    name = "vision"

    def __init__(self, judge: VisionJudge, consecutive: int = 2):
        self.judge = judge
        self.consecutive = consecutive

    def check(self, obs: Observation) -> list[Signal]:
        sheets = [f for f in obs.frames if f.get("sheet")]
        if len(sheets) < 2:
            return []
        reference = sheets[0]["sheet"]
        recent = sheets[1:][-self.consecutive:]
        verdicts = [self.judge.judge(obs.run_dir, row, reference, obs.env_id) for row in recent]
        last = verdicts[-1]
        step = last.step
        ev = {"sheet": last.sheet, "reference": reference, "model": self.judge.model}
        if not last.ok:
            return [Signal("vision", "vision_unavailable", "note", 0.0,
                           f"Could not look at the robot: {last.error[:120]}", step, obs.now, ev)]
        desc = f"posture={last.posture}, motion={last.motion}. Gemma: \"{last.why}\""
        if len(verdicts) >= self.consecutive and all(v.wrong for v in verdicts):
            return [Signal("vision", "robot_looks_wrong", "wake", 1.0,
                           f"The robot looks wrong in the last {self.consecutive} test runs: {desc}",
                           step, obs.now, ev)]
        if last.wrong:
            return [Signal("vision", "robot_looks_wrong", "note", 0.5,
                           f"The robot looked wrong in the latest test run: {desc}", step, obs.now, ev)]
        if last.odd:
            return [Signal("vision", "robot_looks_odd", "note", 0.3,
                           f"The robot looked odd in the latest test run: {desc}", step, obs.now, ev)]
        if not last.robot_visible:
            return [Signal("vision", "robot_out_of_view", "note", 0.3,
                           "The robot is not visible in the latest test run.", step, obs.now, ev)]
        return []


def verdict_json(v: Verdict) -> str:
    return json.dumps(asdict(v))
