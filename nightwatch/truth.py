"""Ground-truth body posture straight from the simulator.

This is what makes the evaluation honest: the watchers only see training metrics and
pictures, while the labels come from the physics state they never get to read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Robots whose torso pitch is a hinge joint at qpos[2] / qvel[2]
PLANAR = ("HalfCheetah", "Hopper", "Walker2d", "Swimmer")
# Robots with a free-floating torso (quaternion at qpos[3:7])
FREE = ("Ant", "Humanoid")


def env_family(env_id: str) -> str:
    for name in PLANAR:
        if env_id.startswith(name):
            return "planar"
    for name in FREE:
        if env_id.startswith(name):
            return "free"
    return "unknown"


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Pose:
    tilt_deg: float          # 0 = upright, 180 = upside down
    ang_rate: float          # rad/s, how fast the torso is rotating
    forward_vel: float       # m/s along x
    max_abs_qvel: float
    finite: bool

    @property
    def upright(self) -> bool:
        return self.finite and self.tilt_deg < 60.0

    @property
    def flipped(self) -> bool:
        return self.finite and self.tilt_deg > 120.0


def pose_from_state(env_id: str, qpos: Any, qvel: Any) -> Pose:
    """Compute torso tilt, spin and speed from MuJoCo qpos/qvel arrays."""
    qp = [float(x) for x in qpos]
    qv = [float(x) for x in qvel]
    finite = all(math.isfinite(x) for x in qp + qv)
    max_abs_qvel = max((abs(x) for x in qv), default=0.0) if finite else float("inf")
    fam = env_family(env_id)
    if not finite:
        return Pose(tilt_deg=float("nan"), ang_rate=float("nan"), forward_vel=float("nan"),
                    max_abs_qvel=max_abs_qvel, finite=False)
    if fam == "planar":
        tilt = abs(math.degrees(wrap_angle(qp[2])))
        return Pose(tilt_deg=tilt, ang_rate=abs(qv[2]), forward_vel=qv[0],
                    max_abs_qvel=max_abs_qvel, finite=True)
    if fam == "free":
        w, x, y, z = qp[3:7]
        up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the torso's up axis
        up_z = max(-1.0, min(1.0, up_z))
        tilt = math.degrees(math.acos(up_z))
        ang_rate = math.sqrt(sum(v * v for v in qv[3:6]))
        return Pose(tilt_deg=tilt, ang_rate=ang_rate, forward_vel=qv[0],
                    max_abs_qvel=max_abs_qvel, finite=True)
    return Pose(tilt_deg=0.0, ang_rate=0.0, forward_vel=qv[0] if qv else 0.0,
                max_abs_qvel=max_abs_qvel, finite=True)


@dataclass
class EpisodeTruth:
    """Summary of one test episode, written to labels/truth.jsonl."""

    poses: list[Pose] = field(default_factory=list)
    warnings: int = 0

    def add(self, pose: Pose) -> None:
        self.poses.append(pose)

    def summary(self) -> dict:
        n = max(len(self.poses), 1)
        finite = [p for p in self.poses if p.finite]
        nonfinite = len(finite) < len(self.poses)
        upright = sum(1 for p in finite if p.upright) / n
        flipped = sum(1 for p in finite if p.flipped) / n
        mean_rate = sum(p.ang_rate for p in finite) / max(len(finite), 1)
        mean_vel = sum(p.forward_vel for p in finite) / max(len(finite), 1)
        max_qvel = max((p.max_abs_qvel for p in self.poses), default=0.0)
        return {
            "upright_frac": round(upright, 3),
            "flipped_frac": round(flipped, 3),
            "mean_ang_rate": round(mean_rate, 3),
            "mean_forward_vel": round(mean_vel, 3),
            "max_abs_qvel": round(max_qvel, 3) if math.isfinite(max_qvel) else "inf",
            "nonfinite": nonfinite,
            "sim_warnings": self.warnings,
        }


# Thresholds that turn a test-episode summary into a label. Kept in one place so the
# post can quote them.
FLIPPED_FRAC = 0.5      # more than half the episode on its back
SPIN_RATE = 6.0         # rad/s average torso rotation = somersaulting, not running
EXPLODED_QVEL = 500.0   # joint speeds no real robot reaches


def label_episode(summary: dict) -> str:
    """'ok' | 'flipped' | 'spinning' | 'exploded'."""
    max_qvel = summary.get("max_abs_qvel", 0)
    if summary.get("nonfinite") or max_qvel == "inf" or (isinstance(max_qvel, (int, float)) and max_qvel > EXPLODED_QVEL):
        return "exploded"
    if summary.get("sim_warnings", 0) > 0 and summary.get("upright_frac", 1) < 0.5:
        return "exploded"
    if summary.get("flipped_frac", 0) >= FLIPPED_FRAC:
        return "flipped"
    if summary.get("mean_ang_rate", 0) >= SPIN_RATE:
        return "spinning"
    return "ok"
