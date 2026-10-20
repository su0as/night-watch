"""End-to-end over synthetic runs: ground truth -> replay -> scores -> EVAL.md."""

import json

from fakes import TruthJudge, make_run
from nightwatch.detectors import PROFILES, build_detectors
from nightwatch.detectors.tabpfn_detector import TabPFNDetector
from nightwatch.detectors.vision import POSTURES, SCHEMA, VisionDetector, VisionJudge
from nightwatch.evaluation.dataset import run_windows
from nightwatch.evaluation.groundtruth import behaviour_failures, failures_for_run, run_span
from nightwatch.evaluation.metrics import score_run, summarise
from nightwatch.evaluation.replay import replay_run
from nightwatch.evaluation.run_eval import auroc
from nightwatch.llm import ChatResult
from nightwatch.policy import PolicyConfig, PolicyState
from nightwatch.report import write_morning_report


def _dataset(tmp_path):
    return {
        "healthy": make_run(tmp_path, "healthy", "healthy", seed=1),
        "lr": make_run(tmp_path, "lr", "lr_blowup", seed=2),
        "flip": make_run(tmp_path, "flip", "flip", seed=3),
        "stall": make_run(tmp_path, "stall", "stall", seed=4),
    }


def test_ground_truth(tmp_path):
    runs = _dataset(tmp_path)
    assert failures_for_run(runs["healthy"]) == []
    assert failures_for_run(runs["lr"])[0].kind == "lr_blowup"
    flip = failures_for_run(runs["flip"])
    assert flip[0].kind == "flipped" and flip[0].source == "behaviour"
    assert failures_for_run(runs["stall"])[0].kind == "stall"
    assert run_span(runs["healthy"]).finished and not run_span(runs["stall"]).finished


def test_behaviour_needs_to_be_sustained():
    rows = [{"step": s, "t": s, "label": lab} for s, lab in
            [(0, "ok"), (10, "flipped"), (20, "ok"), (30, "flipped"), (40, "flipped")]]
    f = behaviour_failures(rows, sustain=2)
    assert f[0].onset_step == 30


def test_windows_are_labelled_after_onset(tmp_path):
    runs = _dataset(tmp_path)
    ws = run_windows(runs["lr"])
    assert {w["label"] for w in ws} == {0, 1}
    assert all(w["label"] == 0 for w in run_windows(runs["healthy"]))


def test_replay_thresholds_vs_vision(tmp_path):
    runs = _dataset(tmp_path)
    cfg = PolicyConfig()
    th = build_detectors(PROFILES["thresholds"])
    vis = build_detectors(PROFILES["vision"], vision_judge=TruthJudge())
    scores = []
    for name, run in runs.items():
        fails = failures_for_run(run)
        span = run_span(run)
        scores.append(score_run(name, "thresholds", fails, replay_run(run, "thresholds", th, cfg).wakes, span))
        scores.append(score_run(name, "vision", fails, replay_run(run, "vision", vis, cfg).wakes, span))
    by = {(s.run_id, s.profile): s for s in scores}
    assert by[("lr", "thresholds")].caught
    assert by[("stall", "thresholds")].caught          # plain rules are in every profile
    assert by[("flip", "thresholds")].caught is False  # the blind spot
    assert by[("flip", "vision")].caught               # eyes on the robot
    assert by[("healthy", "thresholds")].false_wakes == 0
    # Eyes alone miss what only shows in the metrics: you need both, which is the point.
    assert by[("lr", "vision")].caught is False
    summary = {s.profile: s for s in summarise(scores)}
    assert summary["vision"].caught == summary["thresholds"].caught == 2


def test_tabpfn_detector_with_lookup(tmp_path):
    runs = _dataset(tmp_path)
    ws = run_windows(runs["lr"])
    probs = {("lr", w["step"]): (0.95 if w["label"] else 0.05) for w in ws}
    det = TabPFNDetector(lookup=lambda rid, step: probs.get((rid, step)))
    res = replay_run(runs["lr"], "tabpfn", build_detectors(["rules"]) + [det], PolicyConfig())
    assert any(w.source == "tabpfn" for w in res.wakes)
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def chat_json(self, model, prompt, schema, images_b64=None, purpose="", think=False, max_tokens=None):
        self.calls += 1
        self.think = think
        self.n_images = len(images_b64)
        self.prompt = prompt
        assert images_b64
        return self.answer, ChatResult(json.dumps(self.answer), model, 1.5, 900, 40)


def test_vision_judge_schema_cache_and_decision(tmp_path, monkeypatch):
    import nightwatch.detectors.vision as vision_mod

    monkeypatch.setattr(vision_mod, "image_b64", lambda p, max_side=1024: "aW1n")  # no Pillow needed
    run = make_run(tmp_path, "flip", "flip")
    (run / "frames").mkdir(exist_ok=True)
    for line in (run / "frames.jsonl").read_text().splitlines():
        sheet = json.loads(line)["sheet"]
        (run / sheet).write_bytes(b"\x89PNG fake")
    assert set(SCHEMA["properties"]["posture"]["enum"]) == set(POSTURES)
    client = FakeClient({"robot_visible": True, "posture": "upside_down", "motion": "moving_forward",
                         "looks_wrong": True, "why": "legs in the air"})
    judge = VisionJudge(client, model="gemma4:e4b")
    det = VisionDetector(judge)
    from nightwatch.detectors import observe

    obs = observe(run, now=1e12, live=False)
    sigs = det.check(obs)
    assert sigs and sigs[0].level == "wake" and sigs[0].evidence["sheet"].endswith(".png")
    calls = client.calls
    det.check(obs)  # second time comes from the cache
    assert client.calls == calls
    # A model that answers outside the enum is treated as unclear, never as 'wrong'
    weird = VisionJudge(FakeClient({"robot_visible": True, "posture": "dancing", "motion": "spinning?",
                                    "looks_wrong": True, "why": "?"}), model="other")
    v = weird.judge(run, {"step": 5, "sheet": "frames/s000000000_sheet.png"}, "frames/s000000000_sheet.png",
                    "HalfCheetah-v5")
    assert v.posture == "unclear" and not v.wrong
    # Old runs have no full-size frames: the sheet is used, and no reference by default.
    assert client.n_images == 1 and "REFERENCE" not in client.prompt and client.think is False
    # Thinking is part of the cache key: a fast verdict is never reused as a thinking one.
    slow = VisionJudge(FakeClient({"robot_visible": True, "posture": "upright", "motion": "moving_forward",
                                   "why": "fine"}), model="gemma4:e4b", think=True)
    assert slow.cached(run, sigs[0].evidence["sheet"]) is None


def test_vision_uses_full_size_snapshots_and_optional_reference(tmp_path, monkeypatch):
    import nightwatch.detectors.vision as vision_mod

    seen = []
    monkeypatch.setattr(vision_mod, "image_b64", lambda p, max_side=1024: seen.append((str(p), max_side)) or "x")
    row = {"step": 7, "sheet": "frames/s7_sheet.png", "frames": [f"frames/s7_{i}.png" for i in range(4)]}
    client = FakeClient({"robot_visible": True, "posture": "upside_down", "motion": "stuck", "why": "legs up"})
    v = VisionJudge(client, model="m", snapshots=2).judge(tmp_path, row, "frames/ref.png", "HalfCheetah-v5")
    assert v.wrong and client.n_images == 2
    assert [p.replace("\\", "/").split("/")[-1] for p, _ in seen] == ["s7_2.png", "s7_3.png"]
    seen.clear()
    VisionJudge(client, model="m2", snapshots=1, reference=True).judge(tmp_path, row, "frames/ref.png", "HalfCheetah-v5")
    assert client.n_images == 2 and "REFERENCE" in client.prompt
    assert seen[0][0].replace("\\", "/").endswith("frames/ref.png")


class TruncatedClient:
    def chat_json(self, model, prompt, schema, images_b64=None, purpose="", think=False, max_tokens=None):
        return None, ChatResult('{"robot_visible": true, "feet": "abo', model, 99.0, 900, 1500)


def test_vision_truncated_answer_is_not_cached(tmp_path, monkeypatch):
    import nightwatch.detectors.vision as vision_mod

    monkeypatch.setattr(vision_mod, "image_b64", lambda p, max_side=1024: "aW1n")
    run = make_run(tmp_path, "flip", "flip")
    judge = VisionJudge(TruncatedClient(), model="gemma4:e4b")
    v = judge.judge(run, {"step": 5, "sheet": "frames/a.png"}, "frames/b.png", "HalfCheetah-v5")
    assert not v.ok and "unparseable" in v.error
    assert judge.cached(run, "frames/a.png") is None  # retried next check instead of stuck forever


def test_morning_report_without_models(tmp_path, monkeypatch):
    import nightwatch.report as report_mod

    monkeypatch.setattr(report_mod, "timelapse_gif", lambda *a, **k: None)
    monkeypatch.setattr(report_mod, "reward_chart", lambda *a, **k: None)
    run = make_run(tmp_path, "flip", "flip")
    res = replay_run(run, "vision", build_detectors(PROFILES["vision"], vision_judge=TruthJudge()), PolicyConfig())
    st = PolicyState(wakes=res.wakes, notes=res.notes)
    rep = write_morning_report(run, st, client=None)
    md = (run / "report" / "morning.md").read_text()
    assert "Woke you" in md and rep["summary_source"] == "template"


def test_posture_from_floor_is_plain_code():
    from nightwatch.cameras import posture_from_floor

    assert posture_from_floor(["bottom_half"] * 4)[0] == "upright"
    assert posture_from_floor(["top_half", "top_half", "top_half", "bottom_half"])[0] == "upside_down"
    p, m, _ = posture_from_floor(["bottom_half", "top_half", "bottom_half", "top_half"])
    assert m == "spinning"
    assert posture_from_floor(["not_visible"] * 4)[0] == "unclear"
    assert posture_from_floor(["left_or_right"] * 3 + ["bottom_half"])[0] == "on_its_side"


class FloorClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def chat_json(self, model, prompt, schema, images_b64=None, purpose="", think=False, max_tokens=None):
        self.prompts.append(prompt)
        assert len(images_b64) == 1 and "floor" in schema["properties"]
        a = self.answers.pop(0)
        return {"floor": a}, ChatResult(json.dumps({"floor": a}), model, 5.0, 140, 8)


def test_head_camera_judge_asks_one_question_per_snapshot(tmp_path, monkeypatch):
    import nightwatch.detectors.vision as vision_mod

    monkeypatch.setattr(vision_mod, "image_b64", lambda p, max_side=1024: "x")
    row = {"step": 9, "sheet": "frames/s9_sheet.png", "frames": [], "head": [f"frames/s9_head{i}.png" for i in range(4)]}
    client = FloorClient(["top_half", "top_half", "top_half", "not_visible"])
    judge = VisionJudge(client, model="m")
    v = judge.judge(tmp_path, row, "", "HalfCheetah-v5")
    assert v.ok and v.posture == "upside_down" and v.wrong and len(client.prompts) == 4
    assert "floor" in v.why and v.prompt_version.startswith("head-")
    assert judge.cached(tmp_path, row["sheet"]) is not None  # cached under the head version
    # the side view is still used for runs without head frames, or when asked for
    side = VisionJudge(FakeClient({"robot_visible": True, "posture": "upright", "motion": "stuck", "why": "-"}),
                       model="m", view="side")
    assert side.judge(tmp_path, row, "", "HalfCheetah-v5").prompt_version.startswith("v3")
