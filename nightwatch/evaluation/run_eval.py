"""python -m nightwatch eval --runs runs/dataset --out docs/EVAL.md

Replays every recorded run through every watcher profile and writes the tables for the post.
TabPFN is scored leave-one-run-out: the model that judges a run never saw any window of it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median

from ..config import Settings
from ..detectors import PROFILES, build_detectors
from ..features import FEATURE_NAMES, as_vector
from ..policy import PolicyConfig
from ..runlog import RunReader, list_runs
from .dataset import run_windows
from .groundtruth import failures_for_run, run_span
from .metrics import RunScore, score_run, summarise
from .replay import replay_run, truth_by_step


def auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    # Mann-Whitney U with tie handling
    pairs = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i, n = 0.0, 0, len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return round(u / (len(pos) * len(neg)), 3)


def loo_tabpfn(runs: list[Path], settings: Settings) -> tuple[dict, dict]:
    """Leave-one-run-out probabilities for every window of every run."""
    from ..detectors.tabpfn_detector import WindowModel

    windows = {r.name: run_windows(r) for r in runs}
    lookup: dict[tuple[str, int], float] = {}
    all_y, all_p = [], []
    backend = "?"
    started = time.time()
    for held in runs:
        train_rows = [w for name, ws in windows.items() if name != held.name for w in ws]
        test_rows = windows[held.name]
        if not test_rows or not train_rows:
            continue
        model = WindowModel(settings.tabpfn_backend, settings.tabpfn_version).fit(
            [as_vector(w) for w in train_rows], [w["label"] for w in train_rows])
        backend = model.backend_name
        probs = model.predict_proba([as_vector(w) for w in test_rows])
        for w, p in zip(test_rows, probs):
            lookup[(held.name, int(w["step"]))] = p
            all_y.append(int(w["label"]))
            all_p.append(p)
        print(f"[eval] TabPFN LOO {held.name}: {len(test_rows)} windows, backend={backend}", flush=True)
    info = {"backend": backend, "windows": len(all_y), "positives": sum(all_y), "auroc": auroc(all_y, all_p),
            "seconds": round(time.time() - started, 1), "features": len(FEATURE_NAMES)}
    return lookup, info


class CacheOnlyJudge:
    """Uses cached Gemma verdicts only; never calls the model (fast, reproducible eval)."""

    def __init__(self, settings: Settings):
        from ..detectors.vision import make_judge

        self._inner = make_judge(settings, None)
        self.model = settings.vision_model
        self.think = settings.vision_think

    def judge(self, run_dir, frame_row, reference_sheet, env_id):
        from ..detectors.vision import PROMPT_VERSION, Verdict

        hit = self._inner.cached(run_dir, frame_row["sheet"])
        if hit is not None:
            return hit
        return Verdict(step=int(frame_row["step"]), sheet=frame_row["sheet"], model=self.model,
                       prompt_version=self._inner.version or PROMPT_VERSION, ok=False, think=self.think,
                       error="not in cache (run `nightwatch look` first)")


def vision_accuracy(runs: list[Path], judge) -> dict:
    tp = fp = fn = tn = missing = 0
    lat, toks = [], []
    rows_out = []
    for run in runs:
        reader = RunReader(run)
        truth = truth_by_step(run)
        sheets = [f for f in reader.frames() if f.get("sheet")]
        if len(sheets) < 2:
            continue
        ref = sheets[0]["sheet"]
        env_id = reader.manifest().get("env_id", "")
        for row in sheets[1:]:
            v = judge.judge(run, row, ref, env_id)
            t = truth.get(int(row["step"]))
            if not v.ok or t is None:
                missing += 1
                continue
            actual = t.get("label", "ok") != "ok"
            if v.wrong and actual:
                tp += 1
            elif v.wrong:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1
            if v.latency_s:
                lat.append(v.latency_s)
            if v.input_tokens:
                toks.append(v.input_tokens + v.output_tokens)
            rows_out.append({"run": run.name, "step": v.step, "truth": t.get("label"), "posture": v.posture,
                             "motion": v.motion, "wrong": v.wrong, "why": v.why})
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "unjudged": missing,
            "precision": round(prec, 3) if prec is not None else None,
            "recall": round(rec, 3) if rec is not None else None,
            "median_latency_s": round(median(lat), 2) if lat else None,
            "median_tokens": int(median(toks)) if toks else None, "rows": rows_out}


def _cell(s: RunScore) -> str:
    if s.caught is None:
        return "quiet" if s.false_wakes == 0 else f"{s.false_wakes} false wake(s)"
    if not s.caught:
        extra = f", {s.false_wakes} false" if s.false_wakes else ""
        return f"**missed**{extra}"
    extra = f", {s.false_wakes} false" if s.false_wakes else ""
    return f"woke after {s.minutes_to_wake} min ({s.first_wake_source}){extra}"


def run_evaluation(runs_dir: str, profiles: list[str], out_md: str, settings: Settings,
                   vision_live: bool = False, every_s: float = 120.0, tail_s: float = 3600.0) -> dict:
    runs = list_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no runs found in {runs_dir}")
    cfg = PolicyConfig(max_wakes=settings.max_wakes, snooze_s=settings.snooze_min * 60)
    need_tabpfn = any("tabpfn" in PROFILES[p] for p in profiles)
    need_vision = any("vision" in PROFILES[p] for p in profiles)

    if need_tabpfn and not vision_live:
        # The evaluation reads Gemma's cached answers only; free its memory for TabPFN.
        from ..llm import OllamaClient

        for m in {settings.vision_model, settings.text_model}:
            OllamaClient(settings.ollama_url).unload(m)
    lookup, tab_info = (loo_tabpfn(runs, settings) if need_tabpfn else ({}, {}))
    judge = None
    if need_vision:
        if vision_live:
            from ..detectors.vision import make_judge
            from ..llm import OllamaClient

            judge = make_judge(settings, OllamaClient(settings.ollama_url, keep_alive=settings.ollama_keep_alive))
        else:
            judge = CacheOnlyJudge(settings)

    failures = {r.name: failures_for_run(r) for r in runs}
    spans = {r.name: run_span(r) for r in runs}
    scores: list[RunScore] = []
    for profile in profiles:
        detectors = build_detectors(PROFILES[profile],
                                    tabpfn_lookup=(lambda rid, step: lookup.get((rid, step))) if need_tabpfn else None,
                                    vision_judge=judge)
        for run in runs:
            res = replay_run(run, profile, detectors, cfg, every_s=every_s, tail_s=tail_s)
            scores.append(score_run(run.name, profile, failures[run.name], res.wakes, spans[run.name]))
        print(f"[eval] replayed {len(runs)} runs with profile {profile}", flush=True)

    summary = summarise(scores)
    vis = vision_accuracy(runs, judge) if need_vision else {}

    # ------------------------------------------------------------------ markdown
    L = ["# Night Watch evaluation", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(runs)} recorded training runs in `{runs_dir}`. "
         "Ground truth comes from the simulator's body state and the injected-fault log in `labels/`, "
         "which no watcher reads. Every watcher includes the same plain rules (NaN, stalled log, crash).", "",
         "## Did it wake me when it should have?", "",
         "| watcher | failures caught | median minutes to wake | false wake-ups | false wake-ups per 8 h of training |",
         "|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s.profile} | {s.caught} / {s.failures} | {s.median_minutes if s.median_minutes is not None else '-'} "
                 f"| {s.false_wakes} | {s.false_per_8h} |")
    L += ["", "## Run by run", "",
          "| run | what went wrong (ground truth) | " + " | ".join(profiles) + " |",
          "|---|---|" + "---|" * len(profiles)]
    for run in runs:
        f = failures[run.name]
        what = "nothing" if not f else f"{f[0].kind} at step {f[0].onset_step:,} ({f[0].source})"
        cells = [_cell(next(s for s in scores if s.run_id == run.name and s.profile == p)) for p in profiles]
        L.append(f"| {run.name} | {what} | " + " | ".join(cells) + " |")
    if tab_info:
        L += ["", "## TabPFN on its own (leave one run out)", "",
              f"- backend: `{tab_info['backend']}`",
              f"- windows scored: {tab_info['windows']} ({tab_info['positives']} after something broke)",
              f"- AUROC: {tab_info['auroc']}",
              f"- features per window: {tab_info['features']}; total time {tab_info['seconds']} s"]
        if str(tab_info.get("backend", "")).startswith("sklearn"):
            L.append("- **Warning: this used the sklearn fallback, not TabPFN. Do not quote these numbers as TabPFN.**")
    if vis:
        L += ["", "## Gemma on its own (does the robot look wrong?)", "",
              "| | simulator says wrong | simulator says fine |", "|---|---|---|",
              f"| Gemma says wrong | {vis['tp']} | {vis['fp']} |",
              f"| Gemma says fine | {vis['fn']} | {vis['tn']} |", "",
              f"- precision {vis['precision']}, recall {vis['recall']}, unjudged sheets {vis['unjudged']}",
              f"- model `{settings.vision_model}`, "
              + ("head camera: one 'where are the floor and the sky?' answer per snapshot, posture decided by code; "
                 if settings.vision_view == "head" else
                 f"side camera ({'thinks first' if settings.vision_think else 'answers straight away'}, "
                 f"{settings.vision_snapshots} snapshot(s) at up to {settings.vision_max_side} px); ")
              + f"median latency {vis['median_latency_s']} s, median tokens per look {vis['median_tokens']}"]
    L += ["", "Settings: " + f"check every {every_s:.0f} s in replay, tail {tail_s / 60:.0f} min after a run goes "
          f"silent, max {cfg.max_wakes} wake-ups per run, snooze {cfg.snooze_s / 60:.0f} min per kind."]
    out = Path(out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    results = {"summary": [asdict(s) | {"false_per_8h": s.false_per_8h} for s in summary],
               "scores": [asdict(s) for s in scores], "tabpfn": tab_info,
               "vision": {k: v for k, v in vis.items() if k != "rows"},
               "vision_rows": vis.get("rows", []),
               "failures": {k: [asdict(f) for f in v] for k, v in failures.items()}}
    (out.parent / (out.stem + ".json")).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[eval] wrote {out}", flush=True)
    return results


def look_all(runs_dir: str, settings: Settings) -> dict:
    """Fill Gemma's cache for every test episode of every run (slow and steady: run it overnight)."""
    from ..detectors.vision import make_judge
    from ..llm import OllamaClient

    judge = make_judge(settings, OllamaClient(settings.ollama_url, keep_alive=settings.ollama_keep_alive))
    print(f"[look] {settings.vision_model}, {settings.vision_view} view, "
          f"{judge.head_version if settings.vision_view == 'head' else judge.version}", flush=True)
    done = new = failed = 0
    started = time.time()
    for run in list_runs(runs_dir):
        reader = RunReader(run)
        sheets = [f for f in reader.frames() if f.get("sheet")]
        if len(sheets) < 2:
            continue
        env_id = reader.manifest().get("env_id", "")
        for row in sheets[1:]:
            if judge.cached(run, row["sheet"]) is not None:
                done += 1
                continue
            v = judge.judge(run, row, sheets[0]["sheet"], env_id)
            if v.ok:
                new += 1
            else:
                failed += 1
            el = time.time() - started
            print(f"[look] {run.name} step {v.step:,}: {v.posture}/{v.motion} wrong={v.wrong} "
                  f"({v.latency_s}s) {'ERR ' + v.error[:80] if not v.ok else ''} [{new} new, {el / 60:.1f} min]",
                  flush=True)
    return {"cached_before": done, "new": new, "failed": failed, "minutes": round((time.time() - started) / 60, 1)}


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
