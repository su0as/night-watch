import math

import pytest

from fakes import DT, T0, make_run
from nightwatch.detectors import Rules, Thresholds, observe
from nightwatch.detectors.base import Signal
from nightwatch.faults import Fault, parse_fault, reward_bug_bonus
from nightwatch.features import FEATURE_NAMES, as_vector, window_features
from nightwatch.policy import PolicyConfig, PolicyState, decide
from nightwatch.report import numbers_ok, template_summary
from nightwatch.truth import label_episode, pose_from_state


# ------------------------------------------------------------------ faults & truth

def test_parse_fault_variants():
    assert parse_fault("none").kind == "none"
    f = parse_fault("lr_blowup@200000")
    assert (f.kind, f.at_step, f.params["factor"]) == ("lr_blowup", 200000, 200.0)
    f = parse_fault("stall@250000:hold_s=0")
    assert f.params["hold_s"] == 0.0
    with pytest.raises(ValueError):
        Fault(kind="gremlins")


def test_flip_bug_rewards_being_upside_down():
    f = Fault("flip_bug")
    assert reward_bug_bonus(f, 0.0, 0.0) == pytest.approx(0.0)
    assert reward_bug_bonus(f, 180.0, 0.0) == pytest.approx(2.0)
    s = Fault("spin_bug")
    assert reward_bug_bonus(s, 0, 100.0) == pytest.approx(0.5 * 15.0)


def test_pose_and_labels():
    upright = pose_from_state("HalfCheetah-v5", [0, 0, 0.1] + [0] * 6, [1, 0, 0.5] + [0] * 6)
    assert upright.upright and not upright.flipped
    flipped = pose_from_state("HalfCheetah-v5", [0, 0, math.pi] + [0] * 6, [1, 0, 0] + [0] * 6)
    assert flipped.flipped
    # Ant: quaternion for 180 degrees about x -> upside down
    ant = pose_from_state("Ant-v5", [0, 0, 0.5, 0, 1, 0, 0] + [0] * 8, [0] * 14)
    assert ant.flipped
    assert label_episode({"flipped_frac": 0.8}) == "flipped"
    assert label_episode({"mean_ang_rate": 9.0}) == "spinning"
    assert label_episode({"max_abs_qvel": "inf"}) == "exploded"
    assert label_episode({"flipped_frac": 0.0, "mean_ang_rate": 1.0}) == "ok"


# ------------------------------------------------------------------ features

def test_window_features_shape_and_nan_tolerance():
    rows = [{"step": i * 100, "t": i, "rollout/ep_rew_mean": float(i), "train/approx_kl": 0.01} for i in range(1, 20)]
    f = window_features(rows)
    assert set(f) == set(FEATURE_NAMES)
    assert len(as_vector(f)) == len(FEATURE_NAMES)
    assert f["rew_slope_k"] > 0
    rows[-1]["train/value_loss"] = float("nan")
    f2 = window_features(rows)
    assert f2["nonfinite_k"] == 1.0
    assert all(math.isnan(v) or math.isfinite(v) for v in f2.values())


def test_reward_drop_feature(tmp_path):
    run = make_run(tmp_path, "lr", "lr_blowup")
    obs = observe(run, now=T0 + 40 * DT, live=False)
    f = window_features(obs.metrics)
    assert f["rew_drop_from_max"] > 0.5
    assert f["lr_ratio"] == pytest.approx(200.0)


# ------------------------------------------------------------------ rules & thresholds

def test_rules_catch_stall_only_after_silence(tmp_path):
    run = make_run(tmp_path, "stall", "stall", at_row=30)
    rules = Rules(min_stall_s=600)
    just_after = observe(run, now=T0 + 30 * DT + 60, live=False)
    assert not [s for s in rules.check(just_after) if s.kind == "stalled"]
    later = observe(run, now=T0 + 30 * DT + 700, live=False)
    assert [s for s in rules.check(later) if s.kind == "stalled" and s.level == "wake"]


def test_rules_nan_and_finished(tmp_path):
    run = make_run(tmp_path, "nan", "nan", at_row=20)
    obs = observe(run, now=T0 + 25 * DT, live=False)
    assert any(s.kind == "nan" for s in Rules().check(obs))
    healthy = make_run(tmp_path, "ok", "healthy", rows=20)
    end = observe(healthy, now=T0 + 10_000, live=False)
    kinds = {s.kind for s in Rules().check(end)}
    assert kinds == {"finished"}


def test_watcher_never_sees_labels(tmp_path):
    from nightwatch.runlog import RunReader

    run = make_run(tmp_path, "flip", "flip")
    m = RunReader(run).manifest()
    assert "fault" not in m and m["host"] == "test-host"
    obs = observe(run, now=1e12, live=False)
    assert all("label" not in f and "flipped_frac" not in f for f in obs.frames)


def test_live_observe_detects_dead_process(tmp_path):
    import platform

    from nightwatch.runlog import read_json, write_json_atomic

    run = make_run(tmp_path, "crash", "crash")
    m = read_json(run / "manifest.json")
    m["host"] = platform.node()
    write_json_atomic(run / "manifest.json", m)
    obs = observe(run, now=T0 + 30 * DT + 3600, live=True)
    assert obs.process_alive is False  # pid 999999 does not exist
    assert any(s.kind == "dead" for s in Rules().check(obs))


def test_finished_status_hidden_until_end_in_replay(tmp_path):
    run = make_run(tmp_path, "ok", "healthy", rows=20)
    mid = observe(run, now=T0 + 5 * DT, live=False)
    assert mid.status == "running" and len(mid.metrics) == 5


def test_thresholds_catch_lr_blowup_but_not_flip(tmp_path):
    th = Thresholds()
    lr = make_run(tmp_path, "lr", "lr_blowup")
    assert th.check(observe(lr, now=T0 + 35 * DT, live=False))
    flip = make_run(tmp_path, "flip", "flip")
    fired = [s for k in range(12, 61) for s in th.check(observe(flip, now=T0 + k * DT, live=False))]
    assert fired == []  # the blind spot: reward goes up, nothing trips


def test_thresholds_quiet_on_noisy_negative_early_rewards():
    from nightwatch.detectors.base import Observation

    rews = [-300, -280, -260, -200, -150, -120, -100, -140, -110, -130, -105, -125, -115]
    rows = [{"t": float(i), "step": i * 2048, "rollout/ep_rew_mean": r, "train/approx_kl": 0.01}
            for i, r in enumerate(rews)]
    obs = Observation("r", "r", 99.0, False, {"status": "running"}, rows, [], None, None)
    assert Thresholds().check(obs) == []  # old "30% below best" rule would have fired at -140
    rows.append({"t": 99.0, "step": 99 * 2048, "rollout/ep_rew_mean": -280.0, "train/approx_kl": 0.01})
    assert [s.kind for s in Thresholds().check(obs)] == ["reward_drop"]


# ------------------------------------------------------------------ policy

def _sig(kind, level="wake", score=1.0):
    return Signal("test", kind, level, score, f"{kind} happened", 100, 0.0)


def test_policy_snooze_and_cap():
    cfg = PolicyConfig(max_wakes=2, snooze_s=3600)
    st = PolicyState()
    sent, st = decide(st, [_sig("nan")], 1000.0, cfg)
    assert len(sent) == 1
    sent, st = decide(st, [_sig("nan")], 1300.0, cfg)  # same kind inside snooze
    assert sent == [] and st.suppressed == 1 and len(st.notes) == 1
    sent, st = decide(st, [_sig("stalled")], 1400.0, cfg)
    assert len(sent) == 1
    sent, st = decide(st, [_sig("robot_looks_wrong")], 1500.0, cfg)  # cap reached
    assert sent == [] and len(st.wakes) == 2
    sent, st = decide(st, [_sig("drift", level="note")], 1600.0, cfg)
    assert sent == [] and any(n.kind == "drift" for n in st.notes)


def test_policy_strongest_first():
    cfg = PolicyConfig(max_wakes=1)
    sent, _ = decide(PolicyState(), [_sig("weak", score=0.2), _sig("strong", score=0.9)], 0.0, cfg)
    assert [a.kind for a in sent] == ["strong"]


# ------------------------------------------------------------------ report guard

def test_number_guard():
    facts = {"steps_end": 400000, "reward_end": 1234, "watch_from": "23:10", "hours": 7.5}
    assert numbers_ok("Reward ended at 1,234 after 400,000 steps, from 23:10.", facts)[0]
    ok, extra = numbers_ok("Reward ended at 1,500.", facts)
    assert not ok and "1500" in extra
    assert "Quiet night" in template_summary({"run": "r", "watch_from": "a", "watch_to": "b", "reward_start": None,
                                              "wakes": 0, "notes": 0})


# ------------------------------------------------------------------ laptop helpers

def test_keep_awake_never_raises(monkeypatch):
    from nightwatch.keepawake import keep_awake

    monkeypatch.setenv("NW_ALLOW_SLEEP", "1")
    with keep_awake() as ok:
        assert ok is False
    monkeypatch.delenv("NW_ALLOW_SLEEP")
    with keep_awake() as ok:
        assert ok in (True, False)


def test_threads_split_between_parallel_runs(monkeypatch):
    import nightwatch.experiments as ex

    monkeypatch.setattr(ex, "physical_cores", lambda: 4)
    assert ex.threads_per_job(3) == 1
    assert ex.threads_per_job(2) == 2
    assert ex.threads_per_job(1) == 4
    assert ex.threads_per_job(8) == 1
    assert ex.threads_per_job(3, threads=2) == 2


def test_eyetest_scoring():
    from nightwatch.evaluation.eyetest import summarise_mode, truth_class

    assert truth_class(5) == "upright" and truth_class(170) == "upside_down" and truth_class(90) == "on_its_side"
    rows = [
        {"truth": "upright", "posture": "upright", "correct_wake": True, "exact": True, "latency_s": 40.0,
         "out_tokens": 400, "ok": True},
        {"truth": "upside_down", "posture": "upright", "correct_wake": False, "exact": False, "latency_s": 50.0,
         "out_tokens": 500, "ok": True},
        {"truth": "upside_down", "posture": "unclear", "correct_wake": False, "exact": False, "latency_s": 0.0,
         "out_tokens": 0, "ok": False},
    ]
    s = summarise_mode(rows)
    assert s["answered"] == 2 and s["wake_decision_correct"] == 1 and s["median_latency_s"] == 45.0
    assert s["per_truth"]["upside_down"]["said"] == {"upright": 1}


def test_settings_vision_think_default(monkeypatch):
    from nightwatch.config import Settings

    monkeypatch.delenv("NW_VISION_THINK", raising=False)
    s = Settings.from_env(dotenv=None)
    assert s.vision_think is False and s.vision_snapshots == 2 and s.vision_max_side == 1024
    monkeypatch.setenv("NW_VISION_THINK", "1")
    assert Settings.from_env(dotenv=None).vision_think is True
