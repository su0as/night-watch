"""Gemma looks at the robot.

Every few thousand training steps the trainer runs a short test episode and saves four
snapshots as one 2x2 contact sheet. Gemma gets two images: the very first sheet of the
run (the robot standing the right way up, before it learned anything) as a reference,
and the newest sheet. It answers in a fixed JSON schema. The wake decision uses only the
enum fields (posture, motion), never the free-text "why", so a chatty model cannot talk
itself into waking you.

Verdicts are cached per sheet in vision_cache.jsonl, so replays and re-checks are free.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..llm import OllamaClient, OllamaError, image_b64
from ..runlog import append_jsonl, read_jsonl
from .base import Observation, Signal

PROMPT_VERSION = "v1"

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
        "posture": {"type": "string", "enum": POSTURES},
        "motion": {"type": "string", "enum": MOTIONS},
        "looks_wrong": {"type": "boolean"},
        "why": {"type": "string"},
    },
    "required": ["robot_visible", "posture", "motion", "looks_wrong", "why"],
}

ROBOTS = {
    "HalfCheetah": ("a two-legged cheetah robot seen from the side. Its body is a long horizontal "
                    "bar with the head at one end; its two legs hang below the body and touch the floor",
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

Image 1 is a REFERENCE taken at the very start of training: this is the robot the right way up.
Image 2 shows four snapshots (left to right, top to bottom) from the robot's latest test run.

Compare the body in image 2 with image 1 and describe only what you can see:
- posture: is the body the right way up like the reference, upside_down (back on the floor, legs in the air),
  on_its_side, mid_flip, broken (body parts in impossible places, flying apart), or unclear?
- motion across the four snapshots: moving_forward, moving_backward, spinning (rotating/somersaulting),
  stuck, jittering, or unclear?
- looks_wrong: would the owner want to be woken up to see this?
- why: at most 15 words.
"""


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

    def __init__(self, client: OllamaClient, model: str = "gemma4:e4b"):
        self.client = client
        self.model = model

    def _cache_key(self, sheet: str) -> tuple[str, str, str]:
        return (sheet, self.model, PROMPT_VERSION)

    def cached(self, run_dir: str | Path, sheet: str) -> Verdict | None:
        for row in reversed(read_jsonl(Path(run_dir) / "vision_cache.jsonl")):
            if (row.get("sheet"), row.get("model"), row.get("prompt_version")) == self._cache_key(sheet):
                return Verdict.from_dict(row)
        return None

    def judge(self, run_dir: str | Path, frame_row: dict, reference_sheet: str, env_id: str) -> Verdict:
        sheet = frame_row["sheet"]
        hit = self.cached(run_dir, sheet)
        if hit is not None:
            return hit
        robot, task = robot_text(env_id)
        prompt = PROMPT.format(robot=robot, task=task)
        root = Path(run_dir)
        try:
            data, res = self.client.chat_json(
                self.model, prompt, SCHEMA,
                images_b64=[image_b64(root / reference_sheet), image_b64(root / sheet)],
                purpose="look_at_robot")
        except OllamaError as exc:
            # Not cached: a transient failure should be retried next time.
            return Verdict(step=int(frame_row["step"]), sheet=sheet, model=self.model,
                           prompt_version=PROMPT_VERSION, ok=False, error=str(exc))
        v = Verdict(step=int(frame_row["step"]), sheet=sheet, model=self.model,
                    prompt_version=PROMPT_VERSION, ok=data is not None,
                    latency_s=round(res.latency_s, 2), input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens)
        if data is not None:
            v.robot_visible = bool(data.get("robot_visible", True))
            v.posture = data.get("posture") if data.get("posture") in POSTURES else "unclear"
            v.motion = data.get("motion") if data.get("motion") in MOTIONS else "unclear"
            v.looks_wrong = bool(data.get("looks_wrong", False))
            v.why = str(data.get("why", ""))[:200]
        else:
            v.error = "unparseable: " + res.text[:200]
        append_jsonl(root / "vision_cache.jsonl", asdict(v))
        return v


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
