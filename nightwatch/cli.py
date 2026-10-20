"""Command line. Run `python -m nightwatch -h` for the list."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from .config import Settings


def _profile_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="nightwatch",
                   choices=["rules", "thresholds", "tabpfn", "vision", "nightwatch"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nightwatch", description="Sleep while your robot learns.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check Python packages, MuJoCo rendering, Ollama, TabPFN, Temporal")
    p.add_argument("--no-render", action="store_true")

    p = sub.add_parser("ping", help="send a test alert to your phone")
    p.add_argument("--image", default=None)

    p = sub.add_parser("train", help="train one robot with PPO and write telemetry")
    p.add_argument("--env", default="HalfCheetah-v5")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fault", default="none", help="none | flip_bug | spin_bug | lr_blowup@STEP | "
                   "physics_blowup@STEP | stall@STEP[:hold_s=N] | crash@STEP")
    p.add_argument("--name", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--preset", default="fast", choices=["fast", "zoo"])
    p.add_argument("--frame-every", type=int, default=20_000)
    p.add_argument("--eval-steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch decides)")
    p.add_argument("--frame-size", default="1024x768",
                   help="snapshot size. Gemma gets ~1 image token per 48x48 px, so small frames blind it")
    p.add_argument("--no-head-camera", action="store_true", help="side camera only (Gemma cannot judge posture then)")
    p.add_argument("--save-model", action="store_true")

    p = sub.add_parser("runs", help="record the evaluation runs (healthy + broken), several at once")
    p.add_argument("--out", default="runs/dataset")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--env", default="HalfCheetah-v5")
    p.add_argument("--preset", default="fast", choices=["fast", "zoo"])
    p.add_argument("--frame-every", type=int, default=20_000)
    p.add_argument("--smoke", action="store_true", help="4 tiny runs to check the pipeline end to end")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--threads", type=int, default=None,
                   help="torch threads per run (default: CPU cores / jobs, at least 1)")

    p = sub.add_parser("fit", help="build TabPFN's table of past runs (models/windows.csv)")
    p.add_argument("--runs", default="runs/dataset")
    p.add_argument("--out", default=None)

    p = sub.add_parser("look", help="let Gemma look at every test episode of every run (fills the cache)")
    p.add_argument("--runs", default="runs/dataset")

    p = sub.add_parser("eyetest", help="controlled check: can Gemma tell which way up the robot is?")
    p.add_argument("--dir", default="runs/eyetest")
    p.add_argument("--env", default="HalfCheetah-v5")
    p.add_argument("--per-class", type=int, default=6)
    p.add_argument("--modes", nargs="*", default=["side", "head"], choices=["side", "side-think", "head"])
    p.add_argument("--mirror", action="store_true", help="keep Gymnasium's reflective floor")
    p.add_argument("--size", default="1024x768", help="snapshot size, e.g. 320x240 to see the old blindness")
    p.add_argument("--sheet", action="store_true", help="show Gemma the 2x2 contact sheet (the old way)")
    p.add_argument("--out", default="docs/EYETEST.md")

    p = sub.add_parser("eval", help="replay recorded runs through each watcher and write EVAL.md")
    p.add_argument("--runs", default="runs/dataset")
    p.add_argument("--profiles", nargs="*", default=["thresholds", "tabpfn", "vision", "nightwatch"])
    p.add_argument("--out", default="docs/EVAL.md")
    p.add_argument("--vision-live", action="store_true", help="call Gemma for uncached sheets during eval")
    p.add_argument("--every", type=float, default=120.0)

    p = sub.add_parser("check", help="one check of a run right now; prints what each detector says")
    p.add_argument("run")
    _profile_arg(p)

    p = sub.add_parser("watch", help="watch a run with a plain loop (no Temporal)")
    p.add_argument("run")
    p.add_argument("--until", default="07:30", help="07:30 | +8h | +30m | ISO time")
    p.add_argument("--every", type=float, default=None)
    _profile_arg(p)

    p = sub.add_parser("worker", help="run the Temporal worker")
    p.add_argument("--warm", default="nightwatch", help="profile to load at start ('' to skip)")

    p = sub.add_parser("start", help="start the durable overnight watch on Temporal")
    p.add_argument("run")
    p.add_argument("--until", default="07:30")
    p.add_argument("--every", type=float, default=None)
    _profile_arg(p)

    for name in ("status", "stop"):
        p = sub.add_parser(name, help=f"{name} the Temporal watch of a run")
        p.add_argument("run")
    p = sub.add_parser("snooze", help="no wake-ups for N minutes (still noted for the morning)")
    p.add_argument("run")
    p.add_argument("minutes", type=float, nargs="?", default=60.0)

    p = sub.add_parser("report", help="write a morning report for a run from a replay (no live watch)")
    p.add_argument("run")
    _profile_arg(p)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    s = Settings.from_env()

    if args.cmd == "doctor":
        from . import doctor

        sys.exit(1 if doctor.run(s, render=not args.no_render) else 0)

    if args.cmd == "ping":
        from .notify import Notifier

        ch = Notifier(s).send("Night Watch: test", "If you can read this on your phone, alerts work.",
                              urgent=True, image=args.image)
        print(f"sent via {', '.join(ch)}")
        return

    if args.cmd == "train":
        from .train import train

        fw, fh = (int(x) for x in args.frame_size.lower().split("x"))
        train(width=fw, height=fh, env_id=args.env, steps=args.steps, seed=args.seed, fault_spec=args.fault, name=args.name,
              runs_dir=args.runs_dir or s.runs_dir, preset=args.preset, frame_every=args.frame_every,
              eval_steps=args.eval_steps, device=args.device, save_model=args.save_model,
              threads=args.threads, head_camera=not args.no_head_camera)
        return

    if args.cmd == "runs":
        from .experiments import DEFAULT_MATRIX, SMOKE_MATRIX, run_matrix

        if args.smoke:
            res = run_matrix(args.out if args.out != "runs/dataset" else "runs/smoke", SMOKE_MATRIX,
                             steps=min(args.steps, 60_000), jobs=args.jobs, env_id=args.env,
                             preset=args.preset, frame_every=10_000, only=args.only, threads=args.threads)
        else:
            res = run_matrix(args.out, DEFAULT_MATRIX, steps=args.steps, jobs=args.jobs, env_id=args.env,
                             preset=args.preset, frame_every=args.frame_every, only=args.only,
                             threads=args.threads)
        print(json.dumps(res, indent=2))
        return

    if args.cmd == "fit":
        from .evaluation.dataset import build_table

        print(json.dumps(build_table(args.runs, args.out or s.tabpfn_train_table), indent=2))
        return

    if args.cmd == "look":
        from .evaluation.run_eval import look_all

        print(json.dumps(look_all(args.runs, s), indent=2))
        return

    if args.cmd == "eyetest":
        from .evaluation.eyetest import run_eyetest

        size = tuple(int(x) for x in args.size.lower().split("x"))
        res = run_eyetest(args.dir, s, args.modes, env_id=args.env, per_class=args.per_class,
                          mirror=args.mirror, out_md=args.out, size=size, sheet_only=args.sheet)
        print(json.dumps({m: {k: v for k, v in r.items() if k != "rows"} for m, r in res.items()}, indent=2))
        print(f"wrote {args.out}")
        return

    if args.cmd == "eval":
        from .evaluation.run_eval import run_evaluation

        run_evaluation(args.runs, args.profiles, args.out, s, vision_live=args.vision_live, every_s=args.every)
        return

    if args.cmd == "check":
        from .runtime import WatchContext

        ctx = WatchContext(s, args.profile)
        signals, status, step = ctx.check(args.run)
        print(f"status={status} step={step:,}")
        for sig in signals:
            print(f"  [{sig.level}] {sig.source}/{sig.kind}: {sig.message}")
        if not signals:
            print("  nothing to report")
        return

    if args.cmd == "watch":
        from .watch import parse_until, watch_plain

        watch_plain(args.run, args.profile, parse_until(args.until), s, every_s=args.every)
        return

    if args.cmd == "worker":
        from .temporal_app.worker import main as worker_main

        worker_main(s, args.warm or None)
        return

    if args.cmd == "start":
        from .temporal_app.worker import start_watch
        from .watch import parse_until

        until = parse_until(args.until)
        wid = asyncio.run(start_watch(s, args.run, args.profile, until, args.every or s.check_every_s))
        print(f"started {wid}; watching until {time.strftime('%H:%M', time.localtime(until))}. "
              f"Temporal UI: http://localhost:8233")
        return

    if args.cmd in ("status", "stop", "snooze"):
        from .temporal_app import worker as w

        if args.cmd == "status":
            print(json.dumps(asyncio.run(w.query_status(s, args.run)), indent=2))
        elif args.cmd == "stop":
            asyncio.run(w.signal(s, args.run, "stop"))
            print("stop sent; the report will be written now")
        else:
            asyncio.run(w.signal(s, args.run, "snooze", args.minutes))
            print(f"snoozed for {args.minutes:.0f} min")
        return

    if args.cmd == "report":
        from .evaluation.replay import Preloaded, check_times
        from .policy import PolicyConfig, PolicyState, decide
        from .report import write_morning_report
        from .runtime import WatchContext

        ctx = WatchContext(s, args.profile)
        pre = Preloaded.load(args.run)
        state = PolicyState()
        cfg = PolicyConfig(max_wakes=s.max_wakes, snooze_s=s.snooze_min * 60)
        detectors = ctx.detectors
        from .detectors import run_detectors

        times = check_times(pre)
        for now in times:
            _, state = decide(state, run_detectors(pre.observe(now), detectors), now, cfg)
        rep = write_morning_report(args.run, state, client=ctx.client, text_model=s.text_model,
                                   watch_started=times[0] if times else None,
                                   watch_ended=times[-1] if times else None, profile=args.profile)
        print(json.dumps({k: v for k, v in rep.items() if k != "facts"}, indent=2))
        return


if __name__ == "__main__":
    main()
