"""Train a MuJoCo robot with PPO and leave a trail a watcher can follow.

Usage:
    python -m nightwatch train --env HalfCheetah-v5 --steps 400000 --seed 1 --name healthy-1
    python -m nightwatch train --fault flip_bug --name flip-1
    python -m nightwatch train --fault lr_blowup@200000 --name lr-1

What gets written (see runlog.py): training metrics every PPO update, a short test episode
every --frame-every steps with four snapshots of the robot, a heartbeat, and in labels/
the hidden ground truth (injected fault, true body orientation) used only for evaluation.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .faults import Fault, parse_fault, reward_bug_bonus
from .runlog import RunPaths, append_jsonl, read_json, write_json_atomic
from .truth import EpisodeTruth, label_episode, pose_from_state


# --------------------------------------------------------------------------- presets

def ppo_preset(name: str) -> dict:
    if name == "zoo":
        # rl-baselines3-zoo hyperparameters for HalfCheetah (slower, learns better)
        import torch.nn as nn

        return dict(
            n_steps=512, batch_size=64, gamma=0.98, learning_rate=2.0633e-05,
            ent_coef=0.000401762, clip_range=0.1, n_epochs=20, gae_lambda=0.92,
            max_grad_norm=0.8, vf_coef=0.58096,
            policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU,
                               net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )
    # "fast": Stable-Baselines3 defaults. Good enough to see learning in a few hundred k steps.
    return dict(n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)


# --------------------------------------------------------------------------- env pieces

@dataclass
class Clock:
    """Shared step counter so reward bugs can switch on at a given training step."""

    step: int = 0


class RewardBugWrapper(gym.Wrapper):
    """Reward shaping: the intended "keep the torso level" term, and/or the buggy terms (see faults.py).

    shaping="level" is the term written correctly: r -= w * (1 - cos(tilt)). Plain PPO on HalfCheetah
    often learns to run on its back by itself (3 of 4 unshaped runs did, on this project's laptop), so
    runs that are meant to be healthy, or to fail only through an injected training fault, use it.
    The flip bug is the same term with the sign flipped."""

    def __init__(self, env: gym.Env, env_id: str, fault: Fault, clock: Clock, shaping: str = "none",
                 shaping_weight: float = 1.0):
        super().__init__(env)
        self.env_id = env_id
        self.fault = fault
        self.clock = clock
        self.shaping = shaping
        self.shaping_weight = float(shaping_weight)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bug = self.fault.is_reward_bug and self.clock.step >= self.fault.at_step
        if bug or self.shaping == "level":
            u = self.env.unwrapped
            pose = pose_from_state(self.env_id, u.data.qpos, u.data.qvel)
            if pose.finite:
                if self.shaping == "level":
                    pen = self.shaping_weight * (1.0 - math.cos(math.radians(pose.tilt_deg)))
                    reward = float(reward) - pen
                    info["shaping_level"] = -pen
                if bug:
                    bonus = reward_bug_bonus(self.fault, pose.tilt_deg, pose.ang_rate)
                    reward = float(reward) + bonus
                    info["shaping_bonus"] = bonus
        return obs, reward, terminated, truncated, info


CAMERA_DISTANCE = {"HalfCheetah": 3.0, "Hopper": 3.5, "Walker2d": 4.0, "Ant": 5.0, "Humanoid": 4.5}


def camera_config(env_id: str) -> dict:
    """Side view, closer than Gymnasium's default so the robot fills more of the frame.

    Gymnasium renders these robots with a *free* camera that stays where it started (it resets
    the camera type on every render), so a robot that learns to run leaves the frame within one
    test episode. aim_camera() below re-points it at the torso before every snapshot.
    """
    dist = next((d for k, d in CAMERA_DISTANCE.items() if env_id.startswith(k)), 4.0)
    return {"distance": dist, "elevation": -12.0, "azimuth": 90.0}


def aim_camera(env: gym.Env) -> None:
    """Point the render camera at the torso (body 1 in all of these models)."""
    u = env.unwrapped
    try:
        viewer = u.mujoco_renderer._get_viewer("rgb_array")
        viewer.cam.lookat[:] = u.data.xpos[1]
    except Exception:  # noqa: BLE001  older/newer gymnasium: fall back to its default camera
        pass


def matte_floor(env: gym.Env) -> None:
    """Turn off the mirror floor in the rendered test episodes (rendering only, physics untouched).

    The Gymnasium robots stand on a reflective checkerboard, so every picture also contains an
    upside-down copy of the robot under it. That is the one thing Night Watch asks Gemma about.
    """
    try:
        env.unwrapped.model.mat_reflectance[:] = 0.0
    except Exception:  # noqa: BLE001
        pass


def sim_warnings(unwrapped) -> int:
    try:
        return int(sum(int(w.number) for w in unwrapped.data.warning))
    except Exception:  # noqa: BLE001
        return 0


def apply_physics_fault(unwrapped, fault: Fault) -> None:
    m = unwrapped.model
    m.opt.timestep *= float(fault.params["timestep_factor"])
    m.actuator_gear[:, 0] *= float(fault.params["gear_factor"])


# --------------------------------------------------------------------------- logging

class JsonlMetrics(KVWriter):
    """SB3 logger output: one JSON line per dump, plus wall-clock time."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, key_values, key_excluded, step: int = 0) -> None:  # noqa: D401
        row: dict = {"t": time.time()}
        row["step"] = int(key_values.get("time/total_timesteps", step) or step)
        for k, v in key_values.items():
            if isinstance(v, (bool, np.bool_)):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
        append_jsonl(self.path, row)

    def close(self) -> None:
        pass


class OneLine(KVWriter):
    """Short progress line instead of SB3's big table."""

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        s = int(key_values.get("time/total_timesteps", step) or step)
        rew = key_values.get("rollout/ep_rew_mean")
        fps = key_values.get("time/fps")
        rew_s = f"{rew:9.1f}" if isinstance(rew, (int, float)) else "      n/a"
        print(f"[train] step {s:>9,}  reward {rew_s}  fps {fps}", flush=True)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- telemetry

def contact_sheet(frames: list[np.ndarray], cols: int = 2, max_width: int = 1024) -> Image.Image:
    """2x2 overview for humans (reports, alerts). Gemma gets the full-size snapshots instead."""
    imgs = [Image.fromarray(f) for f in frames]
    w, h = imgs[0].size
    if cols * w > max_width:
        w, h = max_width // cols, int(h * (max_width // cols) / w)
        imgs = [im.resize((w, h), Image.LANCZOS) for im in imgs]
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


class Telemetry(BaseCallback):
    def __init__(self, paths: RunPaths, env_id: str, eval_env: gym.Env, train_envs: list,
                 fault: Fault, clock: Clock, frame_every: int, eval_steps: int, n_frames: int,
                 seed: int, heartbeat_every_s: float = 5.0):
        super().__init__(verbose=0)
        self.paths = paths
        self.env_id = env_id
        self.eval_env = eval_env
        self.train_envs = train_envs
        self.fault = fault
        self.clock = clock
        self.frame_every = max(int(frame_every), 1)
        self.eval_steps = int(eval_steps)
        self.n_frames = int(n_frames)
        self.seed = seed
        self.heartbeat_every_s = heartbeat_every_s
        self._next_frame = 0
        self._last_hb = 0.0
        self._fault_fired = fault.kind == "none" or fault.is_reward_bug
        self._render_failed = False
        self._captures = 0
        self.cams = None

    def attach_cameras(self, cams) -> None:
        self.cams = cams if cams is not None and cams.ready else None

    # -- lifecycle
    def _on_training_start(self) -> None:
        self._heartbeat(0)
        self._capture()
        self._next_frame = self.frame_every

    def _on_step(self) -> bool:
        n = self.num_timesteps
        self.clock.step = n
        now = time.time()
        if now - self._last_hb >= self.heartbeat_every_s:
            self._heartbeat(n)
        if not self._fault_fired and n >= self.fault.at_step:
            self._fire_fault(n)
        if n >= self._next_frame:
            self._capture()
            while self._next_frame <= n:
                self._next_frame += self.frame_every
        return True

    def _on_training_end(self) -> None:
        self._capture()
        self._heartbeat(self.num_timesteps)

    # -- pieces
    def _heartbeat(self, step: int, status: str = "running") -> None:
        self._last_hb = time.time()
        write_json_atomic(self.paths.heartbeat,
                          {"pid": os.getpid(), "step": int(step), "t": self._last_hb, "status": status})

    def _fault_event(self, step: int) -> None:
        append_jsonl(self.paths.labels_dir / "fault_events.jsonl",
                     {"kind": self.fault.kind, "step": int(step), "t": time.time()})

    def _fire_fault(self, step: int) -> None:
        self._fault_fired = True
        kind = self.fault.kind
        self._fault_event(step)
        print(f"[fault] {kind} at step {step:,}", flush=True)
        if kind == "lr_blowup":
            current = float(self.model.lr_schedule(self.model._current_progress_remaining))
            blown = current * float(self.fault.params["factor"])
            self.model.lr_schedule = lambda _progress, v=blown: v
        elif kind == "physics_blowup":
            for env in self.train_envs + [self.eval_env]:
                apply_physics_fault(env.unwrapped, self.fault)
        elif kind == "stall":
            hold = float(self.fault.params.get("hold_s", 0))
            # Main loop is stuck: no metrics, no heartbeat, but the process is alive.
            if hold <= 0:
                while True:
                    time.sleep(60)
            time.sleep(hold)
            os._exit(3)
        elif kind == "crash":
            # Simulates the OOM killer: no traceback, no 'finished', nothing.
            os._exit(1)

    def _render(self, env: gym.Env) -> np.ndarray | None:
        if self._render_failed:
            return None
        try:
            if self.cams is not None:
                u = env.unwrapped
                return self.cams.side(u.data.qpos, u.data.qvel)
            aim_camera(env)
            frame = env.render()
            return None if frame is None else np.asarray(frame)
        except Exception as exc:  # noqa: BLE001
            self._render_failed = True
            print(f"[telemetry] rendering failed, continuing without frames: {exc}", flush=True)
            return None

    def _capture(self) -> None:
        start = time.time()
        env = self.eval_env
        vecnorm = self.model.get_vec_normalize_env()
        obs, _ = env.reset(seed=self.seed * 1000 + self._captures)
        self._captures += 1
        u = env.unwrapped
        w0 = sim_warnings(u)
        truth = EpisodeTruth()
        wanted = set(int(round(x)) for x in np.linspace(self.eval_steps * 0.25, self.eval_steps - 1, self.n_frames))
        frames: list[np.ndarray] = []
        heads: list[np.ndarray] = []
        ret = 0.0
        for i in range(self.eval_steps):
            o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _ = self.model.predict(o, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ret += float(r)
            truth.add(pose_from_state(self.env_id, u.data.qpos, u.data.qvel))
            if i in wanted:
                f = self._render(env)
                if f is not None:
                    frames.append(f)
                if self.cams is not None and self.cams.has_head:
                    try:
                        h = self.cams.head(u.data.qpos, u.data.qvel)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[telemetry] head camera failed, continuing without it: {exc}", flush=True)
                        self.cams.has_head, h = False, None
                    if h is not None:
                        heads.append(h)
            if terminated or truncated:
                obs, _ = env.reset()
        truth.warnings = sim_warnings(u) - w0

        step = int(self.num_timesteps)
        t = time.time()
        stem = f"s{step:09d}"
        rel_frames: list[str] = []
        sheet_rel = None
        if frames:
            self.paths.frames_dir.mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(frames):
                rel = f"frames/{stem}_{k}.png"
                Image.fromarray(f).save(self.paths.root / rel)
                rel_frames.append(rel)
            sheet_rel = f"frames/{stem}_sheet.png"
            contact_sheet(frames).save(self.paths.root / sheet_rel)
        rel_heads: list[str] = []
        if heads:
            self.paths.frames_dir.mkdir(parents=True, exist_ok=True)
            for k, h in enumerate(heads):
                rel = f"frames/{stem}_head{k}.png"
                Image.fromarray(h).save(self.paths.root / rel)
                rel_heads.append(rel)
        append_jsonl(self.paths.frames_log, {
            "step": step, "t": t, "sheet": sheet_rel, "frames": rel_frames, "head": rel_heads,
            "eval_return": round(ret, 3), "eval_steps": self.eval_steps,
            "capture_s": round(t - start, 3),
        })
        summary = truth.summary()
        summary.update({"step": step, "t": t, "label": label_episode(summary)})
        append_jsonl(self.paths.truth, summary)


# --------------------------------------------------------------------------- main entry

def train(env_id: str = "HalfCheetah-v5", steps: int = 400_000, seed: int = 1,
          fault_spec: str = "none", name: str | None = None, runs_dir: str = "runs",
          preset: str = "fast", frame_every: int = 25_000, eval_steps: int = 300,
          n_frames: int = 4, width: int = 1024, height: int = 768, device: str = "cpu",
          save_model: bool = False, threads: int | None = None, head_camera: bool = True,
          shaping: str = "none", shaping_weight: float = 1.0) -> Path:
    if shaping not in ("none", "level"):
        raise ValueError(f"unknown shaping {shaping!r}; choose none or level")
    fault = parse_fault(fault_spec)
    if threads:
        # A PPO MLP is tiny; several runs at once each grabbing every core just fight each other.
        import torch

        torch.set_num_threads(int(threads))
    run_name = name or f"{env_id.split('-')[0].lower()}-{fault.kind}-s{seed}-{time.strftime('%m%d-%H%M%S')}"
    paths = RunPaths.of(Path(runs_dir) / run_name)
    if paths.manifest.exists():
        raise SystemExit(f"run {paths.root} already exists; pick another --name")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.labels_dir.mkdir(parents=True, exist_ok=True)

    import stable_baselines3

    manifest = {
        "run_id": run_name, "env_id": env_id, "algo": "PPO", "seed": seed, "preset": preset,
        "total_steps": steps, "frame_every": frame_every, "started_at": time.time(), "threads": threads,
        "frame_size": [width, height], "shaping": shaping, "shaping_weight": shaping_weight,
        "status": "running", "pid": os.getpid(), "host": platform.node(),
        "versions": {"python": sys.version.split()[0], "gymnasium": gym.__version__,
                     "stable_baselines3": stable_baselines3.__version__},
    }
    write_json_atomic(paths.manifest, manifest)
    write_json_atomic(paths.fault, fault.to_dict())

    clock = Clock()
    train_envs: list = []

    def make_train_env():
        env = gym.make(env_id)
        env = RewardBugWrapper(env, env_id, fault, clock, shaping, shaping_weight)
        env = Monitor(env)
        train_envs.append(env)
        return env

    venv = VecNormalize(DummyVecEnv([make_train_env]), norm_obs=True, norm_reward=True)
    venv.seed(seed)
    eval_env = RewardBugWrapper(gym.make(env_id, render_mode="rgb_array", width=width, height=height,
                                         default_camera_config=camera_config(env_id)),
                                env_id, fault, clock, shaping, shaping_weight)
    matte_floor(eval_env)

    model = PPO("MlpPolicy", venv, seed=seed, device=device, verbose=0, **ppo_preset(preset))
    model.set_logger(Logger(folder=None, output_formats=[JsonlMetrics(paths.metrics), OneLine()]))
    telemetry = Telemetry(paths, env_id, eval_env, train_envs, fault, clock,
                          frame_every=frame_every, eval_steps=eval_steps, n_frames=n_frames, seed=seed)
    from .cameras import RobotCameras

    cams = RobotCameras(env_id, width=width, height=height)
    if cams.attach(eval_env, head=head_camera):
        telemetry.attach_cameras(cams)
        print(f"[train] cameras: side {width}x{height}"
              + (", head camera on (what Gemma looks at)" if cams.has_head else ", no head camera"), flush=True)
    else:
        print(f"[train] own cameras unavailable ({cams.error}); using Gymnasium's side view only", flush=True)
        cams = None

    print(f"[train] {run_name}: {env_id}, {steps:,} steps, preset={preset}, fault={fault.kind}@{fault.at_step}",
          flush=True)
    from .keepawake import keep_awake

    try:
        with keep_awake() as awake:
            if awake:
                print("[train] asked the OS not to sleep while training", flush=True)
            model.learn(total_timesteps=steps, callback=telemetry, log_interval=1, progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        m = read_json(paths.manifest, {}) or {}
        m.update({"status": "error", "ended_at": time.time(),
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback_tail": traceback.format_exc().splitlines()[-6:]})
        write_json_atomic(paths.manifest, m)
        write_json_atomic(paths.heartbeat, {"pid": os.getpid(), "step": telemetry.num_timesteps,
                                            "t": time.time(), "status": "error"})
        print(f"[train] crashed: {exc}", flush=True)
        raise
    finally:
        if cams is not None:
            cams.close()
        eval_env.close()
        venv.close()

    if save_model:
        model.save(str(paths.root / "model.zip"))
    m = read_json(paths.manifest, {}) or {}
    m.update({"status": "finished", "ended_at": time.time()})
    write_json_atomic(paths.manifest, m)
    write_json_atomic(paths.heartbeat, {"pid": os.getpid(), "step": int(model.num_timesteps),
                                        "t": time.time(), "status": "finished"})
    print(f"[train] done: {paths.root}", flush=True)
    return paths.root
