"""Synthetic runs with known failures, so the watch/eval pipeline can be tested without
MuJoCo, PyTorch, TabPFN or Ollama."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from nightwatch.runlog import RunPaths, append_jsonl, write_json_atomic

T0 = 1_760_000_000.0
DT = 10.0           # seconds between log rows
STEPS_PER_ROW = 2048
FRAME_EVERY_ROWS = 5


def make_run(root: Path, name: str, scenario: str, rows: int = 60, at_row: int = 30, seed: int = 0,
             finished: bool | None = None) -> Path:
    """scenario: healthy | lr_blowup | flip | stall | crash | nan"""
    rng = random.Random(seed)
    paths = RunPaths.of(root / name)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    silent = scenario in ("stall", "crash")
    n_rows = at_row if silent else rows
    finished = (not silent) if finished is None else finished
    end_t = T0 + n_rows * DT

    fault_kind = {"lr_blowup": "lr_blowup", "flip": "flip_bug", "stall": "stall", "crash": "crash"}.get(scenario, "none")
    write_json_atomic(paths.fault, {"kind": fault_kind, "at_step": at_row * STEPS_PER_ROW, "params": {}})
    if fault_kind in ("lr_blowup", "stall", "crash"):
        append_jsonl(paths.labels_dir / "fault_events.jsonl",
                     {"kind": fault_kind, "step": at_row * STEPS_PER_ROW, "t": T0 + at_row * DT})

    manifest = {"run_id": name, "env_id": "HalfCheetah-v5", "algo": "PPO", "seed": seed,
                "started_at": T0, "status": "finished" if finished else "running", "host": "test-host"}
    if finished:
        manifest["ended_at"] = end_t + 1
    write_json_atomic(paths.manifest, manifest)

    for i in range(1, n_rows + 1):
        t = T0 + i * DT
        step = i * STEPS_PER_ROW
        progress = i / rows
        rew = 1500 * (1 - math.exp(-3 * progress)) + rng.gauss(0, 25)
        kl = 0.012 + abs(rng.gauss(0, 0.003))
        vloss = 0.5 + abs(rng.gauss(0, 0.05))
        std = 1.0 - 0.4 * progress
        lr = 3e-4
        if scenario == "lr_blowup" and i > at_row:
            k = i - at_row
            rew = 1500 * (1 - math.exp(-3 * at_row / rows)) - 250 * k + rng.gauss(0, 25)
            kl = 0.4 + 0.05 * k
            vloss = 0.5 * (10 ** min(k, 4))
            lr = 3e-4 * 200
        if scenario == "flip" and i > at_row - 10:
            rew += 40 * (i - at_row + 10)  # reward keeps climbing, faster than healthy
        row = {"t": t, "step": step, "rollout/ep_rew_mean": rew, "rollout/ep_len_mean": 1000.0,
               "train/approx_kl": kl, "train/clip_fraction": 0.1, "train/entropy_loss": -8 + 2 * progress,
               "train/value_loss": vloss, "train/explained_variance": 0.9, "train/std": std,
               "time/fps": 900.0 + rng.gauss(0, 20), "train/learning_rate": lr}
        if scenario == "nan" and i >= at_row:
            row["train/value_loss"] = float("nan")
        append_jsonl(paths.metrics, row)

        if i % FRAME_EVERY_ROWS == 0 or i == 1:
            f_step = step if i > 1 else 0
            label = "ok"
            flipped = 0.0
            if scenario == "flip" and i >= at_row:
                label, flipped = "flipped", 0.9
            append_jsonl(paths.frames_log, {"step": f_step, "t": t, "sheet": f"frames/s{f_step:09d}_sheet.png",
                                            "frames": [], "eval_return": rew})
            append_jsonl(paths.truth, {"step": f_step, "t": t, "upright_frac": 1 - flipped,
                                       "flipped_frac": flipped, "mean_ang_rate": 1.0, "mean_forward_vel": 2.0,
                                       "max_abs_qvel": 20.0, "nonfinite": False, "sim_warnings": 0,
                                       "label": label})
    write_json_atomic(paths.heartbeat, {"pid": 999999, "step": n_rows * STEPS_PER_ROW, "t": end_t,
                                        "status": "finished" if finished else "running"})
    return paths.root


class TruthJudge:
    """A stand-in for Gemma that answers from the hidden truth file (perfect eyesight).
    Used only to test the plumbing around the vision detector."""

    model = "fake-eyes"

    def judge(self, run_dir, frame_row, reference_sheet, env_id):
        from nightwatch.detectors.vision import PROMPT_VERSION, Verdict

        truth = {}
        for line in (Path(run_dir) / "labels" / "truth.jsonl").read_text().splitlines():
            r = json.loads(line)
            truth[int(r["step"])] = r
        t = truth.get(int(frame_row["step"]), {})
        wrong = t.get("label", "ok") != "ok"
        return Verdict(step=int(frame_row["step"]), sheet=frame_row["sheet"], model=self.model,
                       prompt_version=PROMPT_VERSION, ok=True, robot_visible=True,
                       posture="upside_down" if wrong else "upright",
                       motion="moving_forward", looks_wrong=wrong, why="test")
