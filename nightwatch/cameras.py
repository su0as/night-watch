"""The trainer's cameras: the usual side view, plus a head camera bolted to the robot.

Why a head camera: from the side camera, Gemma 4 E4B could not tell an upside-down
HalfCheetah from an upright one. At any picture size and with any wording it gave the same
answer for both (see docs/EYETEST.md). From a camera fixed to the front of the robot's body,
the question becomes "where is the floor in this picture?", which it gets right in about 5 s.
Upright, the floor fills the bottom half; on its back the camera is upside down too and the
floor fills the top half; spinning, the floor moves around between snapshots.

Both views come from one renderer on a copy of the model (built with MjSpec) that has the
extra camera and a matte floor. The physics the policy trains on is untouched; each render
copies qpos/qvel from the live simulation. One renderer, one OpenGL context: mixing this with
Gymnasium's own renderer made the side frames come out half black on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass

# Where to bolt the head camera, in the torso's own frame (x forward, z up when upright).
# It looks straight ahead along +x.
MOUNTS = {
    "HalfCheetah": {"body": "torso", "pos": [0.55, 0.0, 0.05], "fovy": 75.0},
    "Hopper": {"body": "torso", "pos": [0.08, 0.0, 0.15], "fovy": 75.0},
    "Walker2d": {"body": "torso", "pos": [0.08, 0.0, 0.15], "fovy": 75.0},
    "Ant": {"body": "torso", "pos": [0.3, 0.0, 0.0], "fovy": 75.0},
    "Humanoid": {"body": "head", "pos": [0.1, 0.0, 0.0], "fovy": 75.0},
}
SIDE_DISTANCE = {"HalfCheetah": 3.0, "Hopper": 3.5, "Walker2d": 4.0, "Ant": 5.0, "Humanoid": 4.5}

FLOOR_POSITIONS = ["bottom_half", "top_half", "left_or_right", "not_visible"]


def mount_for(env_id: str) -> dict | None:
    for key, val in MOUNTS.items():
        if env_id.startswith(key):
            return val
    return None


@dataclass
class RobotCameras:
    """side(qpos, qvel) and head(qpos, qvel) -> RGB arrays, from one offscreen renderer."""

    env_id: str
    width: int = 1024
    height: int = 768
    head_width: int = 512
    head_height: int = 384

    def __post_init__(self) -> None:
        self._renderer = None
        self._model = None
        self._data = None
        self._side = None
        self.has_head = False
        self.error = ""

    def attach(self, env, head: bool = True, matte: bool = True) -> bool:
        try:
            import mujoco

            u = env.unwrapped
            spec = mujoco.MjSpec.from_file(u.fullpath)
            mount = mount_for(self.env_id) if head else None
            if mount is not None:
                cam = spec.body(mount["body"]).add_camera()
                cam.name = "nightwatch_head"
                cam.pos = list(mount["pos"])
                cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
                cam.alt.xyaxes = [0, -1, 0, 0, 0, 1]  # image right = robot's right, image up = robot's up
                cam.fovy = float(mount["fovy"])
            if matte:
                for mat in spec.materials:
                    mat.reflectance = 0.0  # no mirror image of the robot under the robot
            spec.visual.global_.offwidth = max(self.width, self.head_width)
            spec.visual.global_.offheight = max(self.height, self.head_height)
            self._model = spec.compile()
            if self._model.nq != u.model.nq or self._model.nv != u.model.nv:
                raise RuntimeError("render model does not match the training model")
            self._data = mujoco.MjData(self._model)
            self._renderer = mujoco.Renderer(self._model, self.height, self.width)
            self._side = mujoco.MjvCamera()
            self._side.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._side.distance = next((d for k, d in SIDE_DISTANCE.items() if self.env_id.startswith(k)), 4.0)
            self._side.elevation = -12.0
            self._side.azimuth = 90.0
            self.has_head = mount is not None
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._renderer = None
            return False

    @property
    def ready(self) -> bool:
        return self._renderer is not None

    def _set_state(self, qpos, qvel) -> None:
        import mujoco

        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)

    def side(self, qpos, qvel):
        if self._renderer is None:
            return None
        self._set_state(qpos, qvel)
        self._side.lookat[:] = self._data.xpos[1]  # follow the torso so a running robot stays in frame
        self._renderer.update_scene(self._data, camera=self._side)
        return self._renderer.render().copy()

    def head(self, qpos, qvel):
        if self._renderer is None or not self.has_head:
            return None
        import numpy as np
        from PIL import Image

        self._set_state(qpos, qvel)
        self._renderer.update_scene(self._data, camera="nightwatch_head")
        img = Image.fromarray(self._renderer.render())
        return np.asarray(img.resize((self.head_width, self.head_height), Image.LANCZOS))

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None


def posture_from_floor(floors: list[str]) -> tuple[str, str, str]:
    """Turn Gemma's per-snapshot floor answers into (posture, motion, feet). Pure code, no model.

    The model only answers a perceptual question; this function makes the call, so the same
    answers always lead to the same decision.
    """
    seen = [f for f in floors if f in FLOOR_POSITIONS and f != "not_visible"]
    if not seen:
        return "unclear", "unclear", "unclear"
    top = sum(f == "top_half" for f in seen)
    bottom = sum(f == "bottom_half" for f in seen)
    side = sum(f == "left_or_right" for f in seen)
    changes = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
    motion = "spinning" if (top and bottom and changes >= 2) else "unclear"
    n = len(seen)
    if top > n / 2:
        return "upside_down", motion, "above_body"
    if bottom > n / 2:
        return "upright", motion, "below_body"
    if side > n / 2:
        return "on_its_side", motion, "unclear"
    return "mid_flip", motion, "mixed"
