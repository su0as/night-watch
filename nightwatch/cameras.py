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
# It looks straight ahead along +x. It sits at the centre of the torso, not on the nose: on the
# nose it went through the floor whenever the robot learned a nose-down crouch, and rendered the
# underside of the floor. The head camera does not draw the robot itself (its own body blocked
# the view), only the world: floor and sky. Think of it as an artificial horizon.
MOUNTS = {
    "HalfCheetah": {"body": "torso", "pos": [0.0, 0.0, 0.0], "fovy": 75.0},
    "Hopper": {"body": "torso", "pos": [0.0, 0.0, 0.0], "fovy": 75.0},
    "Walker2d": {"body": "torso", "pos": [0.0, 0.0, 0.0], "fovy": 75.0},
    "Ant": {"body": "torso", "pos": [0.0, 0.0, 0.0], "fovy": 75.0},
    "Humanoid": {"body": "torso", "pos": [0.0, 0.0, 0.0], "fovy": 75.0},
}
ROBOT_GROUP = 5  # geom group the robot's own geoms are moved to, hidden from the head camera
SIDE_DISTANCE = {"HalfCheetah": 3.0, "Hopper": 3.5, "Walker2d": 4.0, "Ant": 5.0, "Humanoid": 4.5}

FLOOR_POSITIONS = ["bottom_half", "top_half", "whole_picture", "left_or_right", "not_visible"]
SKY_POSITIONS = FLOOR_POSITIONS
# Floor everywhere (nose-down crouch) or nowhere (rearing up): says nothing about upside down.
AMBIGUOUS = ("whole_picture", "not_visible")


def snapshot_vote(floor: str, sky: str | None = None) -> str:
    """'up' | 'down' | 'side' | '?' for one head-camera snapshot.

    Upside down needs the sky BELOW: on its back, the grey sky fills the bottom of the picture,
    which never happens the right way up. Floor alone was not enough: in a steep nose-down crouch
    the floor fills almost the whole picture and Gemma called that "floor in the top half"."""
    if sky is None:  # older answers: floor only
        return {"bottom_half": "up", "top_half": "down", "left_or_right": "side"}.get(floor, "?")
    if sky == "bottom_half" and floor != "bottom_half":
        return "down"
    if (sky == "top_half" or floor == "bottom_half") and sky != "bottom_half":
        return "up"
    if "left_or_right" in (floor, sky):
        return "side"
    return "?"


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
            # Everything not attached to the world body is the robot: give it its own group so
            # the head camera can leave it out. (Visual only; contacts use contype/conaffinity.)
            world = spec.worldbody
            world_geoms = {id(g) for g in world.geoms}
            for g in spec.geoms:
                if id(g) not in world_geoms:
                    g.group = ROBOT_GROUP
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
            self._side_opt = mujoco.MjvOption()
            self._side_opt.geomgroup[ROBOT_GROUP] = 1
            self._head_opt = mujoco.MjvOption()
            self._head_opt.geomgroup[ROBOT_GROUP] = 0
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
        self._renderer.update_scene(self._data, camera=self._side, scene_option=self._side_opt)
        return self._renderer.render().copy()

    def head(self, qpos, qvel):
        if self._renderer is None or not self.has_head:
            return None
        import numpy as np
        from PIL import Image

        self._set_state(qpos, qvel)
        self._renderer.update_scene(self._data, camera="nightwatch_head", scene_option=self._head_opt)
        img = Image.fromarray(self._renderer.render())
        return np.asarray(img.resize((self.head_width, self.head_height), Image.LANCZOS))

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None


def posture_from_votes(votes: list[str]) -> tuple[str, str, str]:
    """Turn per-snapshot votes ('up', 'down', 'side', '?') into (posture, motion, feet).
    Pure code, no model: the same answers always lead to the same decision."""
    seen = [v for v in votes if v in ("up", "down", "side")]
    if not seen:
        return "unclear", "unclear", "unclear"
    down = seen.count("down")
    up = seen.count("up")
    side = seen.count("side")
    changes = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
    motion = "spinning" if (down and up and changes >= 2) else "unclear"
    n = len(seen)
    if down > n / 2:
        return "upside_down", motion, "above_body"
    if up > n / 2:
        return "upright", motion, "below_body"
    if side > n / 2:
        return "on_its_side", motion, "unclear"
    return "mid_flip", motion, "mixed"


def posture_from_floor(floors: list[str]) -> tuple[str, str, str]:
    """Floor-only answers (head-v1/v2 caches and tests)."""
    return posture_from_votes([snapshot_vote(f) if f not in AMBIGUOUS else "?" for f in floors])
