"""Faults we inject on purpose, so we know exactly when something went wrong.

Each one is a failure that has cost someone a night of compute:

    flip_bug        I meant to *penalise* torso tilt to keep the robot level, but the
                    sign is flipped, so tilt is rewarded. The reward curve climbs all
                    night while the robot learns to lie on its back.
    spin_bug        Same mistake with a "stop wobbling" term: rotation gets rewarded,
                    and the robot learns to somersault.
    lr_blowup       Learning rate multiplied mid-run (a bad schedule, a typo in a sweep).
    physics_blowup  Simulator timestep and motor gear raised mid-run, so the physics
                    goes unstable, like a bad URDF or a curriculum step gone wrong.
    stall           The trainer hangs (deadlocked data loader, NFS mount, full disk).
    crash           The process dies without a goodbye (OOM killer, driver crash).

Faults are written to labels/fault.json, which watchers never read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

KINDS = ("none", "flip_bug", "spin_bug", "lr_blowup", "physics_blowup", "stall", "crash")
REWARD_BUGS = ("flip_bug", "spin_bug")
TRAINING_FAULTS = ("lr_blowup", "physics_blowup", "stall", "crash")

DEFAULT_PARAMS: dict[str, dict] = {
    "flip_bug": {"weight": 1.0},
    "spin_bug": {"weight": 0.5, "rate_cap": 15.0},
    "lr_blowup": {"factor": 200.0},
    "physics_blowup": {"timestep_factor": 4.0, "gear_factor": 3.0},
    "stall": {"hold_s": 1800.0},  # 0 = hang forever
    "crash": {},
    "none": {},
}


@dataclass
class Fault:
    kind: str = "none"
    at_step: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown fault {self.kind!r}; choose from {KINDS}")
        merged = dict(DEFAULT_PARAMS.get(self.kind, {}))
        merged.update(self.params or {})
        self.params = merged

    @property
    def is_reward_bug(self) -> bool:
        return self.kind in REWARD_BUGS

    def to_dict(self) -> dict:
        return asdict(self)


def parse_fault(spec: str | None) -> Fault:
    """'none', 'flip_bug', 'lr_blowup@200000', 'stall@250000:hold_s=0'."""
    if not spec or spec == "none":
        return Fault()
    params: dict = {}
    if ":" in spec:
        spec, raw = spec.split(":", 1)
        for part in raw.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = float(v)
    kind, _, at = spec.partition("@")
    return Fault(kind=kind.strip(), at_step=int(float(at)) if at else 0, params=params)


def reward_bug_bonus(fault: Fault, tilt_deg: float, ang_rate: float) -> float:
    """Extra reward that the buggy shaping term hands out (pure function, testable).

    flip_bug: intended  r -= w * (1 - cos(tilt));  bug: r += w * (1 - cos(tilt))
    spin_bug: intended  r -= w * |rate|;           bug: r += w * min(|rate|, cap)
    """
    import math

    if fault.kind == "flip_bug":
        return fault.params["weight"] * (1.0 - math.cos(math.radians(tilt_deg)))
    if fault.kind == "spin_bug":
        return fault.params["weight"] * min(abs(ang_rate), fault.params["rate_cap"])
    return 0.0
