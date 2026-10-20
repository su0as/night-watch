<!--
DRAFT for the DEV post. Do not publish with any [[...]] left in it.
- [[N]]-style brackets = a number you measured (from docs/EVAL.md, the morning report, or your phone).
- [[TRUE? ...]] = a personal detail; keep it only if it is true, otherwise rewrite it.
- Keep it under ~2,500 words. Keep the tone plain. Delete this comment block before publishing.

Title options (pick one after you see the results):
1. My robot's reward went up all night. It had learned to lie on its back.
2. Gemma couldn't tell my robot was upside down. So I gave the robot a head camera.
3. I wanted to sleep while my robot learned to run, so two open models watch it for me
4. Night Watch: TabPFN reads my training logs, Gemma watches my robot, and I sleep

VERIFIED on the Windows laptop (i7-1165G7, 16 GB, Iris Xe), 20 Oct 2026. You can quote these:
- docs/EYETEST.md: gemma4:e4b-it-qat, 12 controlled test episodes (6 upright, 6 upside down).
  Side camera: 6/12 right, 0/6 upside-down caught (it said "upright" every time), both as a 2x2 sheet
  of 320x240 snapshots and as full 1024x768 snapshots. Head camera: 12/12, 6/6, 48/48 floor answers right,
  ~19 s per test episode (4 looks of ~5 s on the iGPU).
- Also tried and failed from the side: other wordings ("what touches the floor", "is the body above the
  feet"), a reference picture, bigger pictures, thinking mode (6 minutes per look on the laptop, still
  wrong on upside-down), gemma4:12b (does not fit the iGPU: ErrorOutOfDeviceMemory).
- Ollama gives Gemma 4 about one image token per 48x48 px: 320x240 = 80 tokens, a 640x480 2x2 sheet = ~130
  for all four snapshots, 1024x768 = 336.
- TabPFN-3.5 local on the laptop CPU, 2,300 training windows: default fit 7 s but every predict ~96 s;
  fit_with_cache 92 s once, then 0.7-0.9 s per live check, same accuracy (0.95 on a synthetic table).
- PPO, 3 runs in parallel on 4 cores: 485 steps/s per run with 4 torch threads each, 610 with 1 thread each.
- The laptop was set to sleep after 10 minutes on AC. Training and the watcher now ask Windows to stay awake.
- Gymnasium's floor is a mirror (reflectance 0.5): every picture also had an upside-down robot in the floor.
- TabPFN's licence prompt crashes on Windows (select() on stdin); a TABPFN_TOKEN works.
- First head-camera mount was on the robot's nose. In the healthy smoke run the robot learned a nose-down
  crouch, the camera went through the floor, rendered the floor's underside, and Gemma said "floor in the top
  half": a false wake of my own making (smoke eval: Gemma precision 0.75, 2 false positives). Fix: camera at
  the torso centre, robot's own geoms hidden from it (an artificial horizon), plus a "floor fills the whole
  picture" answer that counts as no evidence. The eyetest now includes crouch poses so this stays caught.
Cover image: a frame of the cheetah on its back next to the rising reward curve.
-->

---
title: [[pick a title]]
published: false
tags: devchallenge, hf26challenge, ai, robotics
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

## What I Built

I'm doing a master's in robotics, and a lot of robot learning happens in simulation, overnight. You start a run before bed, and then you wake up at 2 a.m. to check it. [[TRUE? adjust to your own habit: how often you check, what you lost to a dead run]]

There are two reasons to check. The boring one: runs die quietly. A hung data loader, a NaN, a crash, and the whole night is wasted. The interesting one: **runs can go wrong while looking great.**

Here is a run I trained this week. I added a reward term to keep the robot's body level and got the sign wrong, so the robot is *paid* for tilting. Look at the reward curve:

<!-- Verified fact you can keep (from the CI smoke run of this repo, 40k steps): reward 709 -> 1244 while a healthy run with the same seed sat around -290; at step 40,000 the simulator says the torso was upside down for 92% of the test episode; the threshold watcher's report said "Nothing worth waking you for". Images: docs/assets/smoke_flip_*.png. Prefer the numbers from your own longer runs if you have them. -->

![reward climbing all night]([[upload reward.png from runs/flip-s1/report]])

It climbs. Every alert rule I would normally set (reward dropped 30%, KL spiked, loss exploded) stays silent, because nothing is dropping. Now look at the robot:

![the robot on its back]([[upload the last contact sheet or timelapse.gif]])

It learned to lie on its back and kick. You only catch that by looking at it.

**Night Watch** looks for me. While a robot trains overnight:

- **Gemma 4** (running locally through Ollama) looks through a camera bolted to the robot's head every 20k training steps and says where the floor is;
- **TabPFN** reads the training metrics and asks whether they look like past runs that broke;
- a few **plain rules** catch NaNs, a silent log and a dead process;
- a **Temporal** workflow keeps the watch alive even if the watcher itself dies at 3 a.m.

It wakes me (a phone notification with the picture attached) only when it's worth it: at most 3 times a night, with no repeats for the same problem within an hour. Everything else goes into a one-page morning report with a time-lapse of the night.

It's for anyone who trains things overnight and would rather sleep than babysit a progress bar.

## Demo

{% embed [[YouTube or Loom link, 60 to 90 s: start the watch, the 3 a.m. phone alert, the morning report]] %}

![time-lapse of one night]([[timelapse.gif]])

What I woke up to: [[screenshot of the phone notification with the contact sheet]]

## Code

{% github su0as/night-watch %}

MIT licensed, started and finished during the challenge. [[N]] tests, including one that runs a full 8-hour night through Temporal's time-skipping test server in a few seconds.

## How I Built It

```plaintext
trainer (PPO, MuJoCo HalfCheetah)              watcher (every 5 min, a Temporal workflow)
  each PPO update   -> metrics.jsonl  ------>   rules    NaN? silent log? process gone?
  every 20k steps   -> 4 snapshots   ------->   TabPFN   do these metrics look like runs that broke?
                       of a test run  ------>   Gemma 4  same robot as the reference picture? upright? spinning?
  heartbeat.json    -> alive?                   policy   wake me (max 3/night) or note it for the morning
  labels/ (hidden)  -> ground truth             report   time-lapse, reward chart, a summary whose numbers are checked
```

### First, a way to know when it is actually right

A watcher that says "looks fine" is useless unless I can check it against the truth. So the trainer writes two kinds of files. The ones the watcher may read are the metrics, the snapshots and a heartbeat. The ones it may not read go in a `labels/` folder: which fault I injected and when, plus the robot's true body angle straight from the physics engine. No watcher touches `labels/`; only the evaluation does. When I say "Gemma caught the flip", that's checked against the simulator, not against Gemma.

Then I broke things on purpose, 12 runs in total:

| run | what I did to it |
|---|---|
| 4 x healthy | nothing (this is where false alarms come from) |
| 2 x flip bug | "keep the body level" reward with the sign flipped |
| 1 x spin bug | "stop wobbling" reward with the sign flipped |
| 2 x LR blow-up | learning rate x200 halfway through |
| 1 x physics blow-up | simulator timestep x4 and motor gear x3 halfway through |
| 1 x stall | the training loop hangs; the process stays alive |
| 1 x crash | the process just disappears |

### Gemma 4 couldn't see it. So the robot got a head camera.

My first version showed Gemma the obvious picture: the side view, four snapshots in a 2x2 grid, plus the robot's first picture as a reference. Before trusting it with a night, I tested its eyesight: drop the robot into a known pose (the right way up, or on its back), let the physics settle it, take four snapshots exactly like a real test episode, and ask the same judge that runs at night. The simulator knows the true answer.

It called every robot "upright". Including the six lying on their backs with their legs in the air.

I tried the usual fixes. Bigger pictures (Ollama gives Gemma about one image token per 48x48 pixels, so each snapshot in my 640x480 grid got ~33 tokens; at 1024x768 it gets 336). Different questions ("what is touching the floor?", "is the body above the feet?"). Letting it think first: 6 minutes per look on my laptop, and still wrong. The 12B model: doesn't fit my laptop's iGPU.

Then I thought about it like a robotics problem instead of a prompt problem. A real robot doesn't work out which way up it is from a photographer standing beside it. It has its own sensors. So I bolted a camera to the robot's head (in a render-only copy of the model, so the physics is untouched). When the robot flips, its camera flips with it, and the question becomes: **where is the floor in this picture?**

![side camera vs head camera, and what Gemma said](https://raw.githubusercontent.com/su0as/night-watch/main/docs/assets/eyetest_side_vs_head.png)

| what Gemma is shown | right calls | upside-down robots caught | seconds per test episode |
|---|---|---|---|
| side camera, 2x2 sheet (my first version) | 6 / 12 | 0 / 6 | 10 |
| side camera, full-size snapshots | 6 / 12 | 0 / 6 | 19 |
| head camera | **12 / 12** | **6 / 6** | 19 |

Gemma answers one question per head-camera snapshot, in a fixed JSON schema (floor in the bottom half, top half, a side, or not visible). **Plain code** turns the four answers into a posture: floor mostly on top means upside down, floor jumping between top and bottom means spinning. Gemma never decides to wake me; it only reports what it sees, and it has to be the same two test episodes in a row.

[[One or two real examples from docs/EVAL.json vision_rows on your real runs: what it said for a flipped run, and anything it got wrong]]

### TabPFN reads the metrics

Every window of 8 log rows becomes one table row: 22 numbers such as how far the reward is below its best, KL divergence relative to its usual level, value-loss ratio and learning-rate ratio. Each row is labelled 1 if the run was already broken at that point. TabPFN learns in context from that table, so there's no training run to babysit (which would have been ironic). I score it leave-one-run-out: the model judging a run has never seen any window from it.

### Plain rules for the obvious

NaN in the log, no new metrics for 6x the usual interval (minimum 10 minutes), or the process gone: those don't need a model, and every watcher in the comparison has them, so the comparison is about what the models add.

### Temporal keeps the watch alive

The watch is a Temporal workflow: check, decide, maybe alert, then wait 5 minutes (a durable timer), and repeat until 07:30. The checks are activities with retries; Gemma loading slowly or Ollama restarting is just a retry. Every 150 checks it continues as new, carrying its state (who woke me, when).

The test that matters: around midnight I closed the worker window. The training kept running, and the workflow sat on the server as "Running" with nobody executing it. [[N]] minutes later I started the worker again and it carried on from its last check, still knowing it had already woken me [[N]] times.

![Temporal UI around the gap]([[screenshot]])

With my first version (a plain `while True: sleep(300)` loop), closing that window just ended the watch, silently. That's the exact failure Night Watch exists to catch, so the watcher can't have it too.

### A morning summary that can't make things up

Gemma writes the three-sentence summary, but only from facts the code computed. A check pulls every number out of its paragraph; if any number isn't in the facts, the paragraph is thrown away and a plain template is used. The report says which one you got. [[TRUE? how often it fell back in your runs]]

### Results

From `docs/EVAL.md` (12 recorded runs, replayed through four watchers):

| watcher | failures caught | median minutes to wake | false wake-ups |
|---|---|---|---|
| alert thresholds (what I used to use) | [[ ]] | [[ ]] | [[ ]] |
| TabPFN on metrics | [[ ]] | [[ ]] | [[ ]] |
| Gemma looking at the robot | [[ ]] | [[ ]] | [[ ]] |
| Night Watch (all of it) | [[ ]] | [[ ]] | [[ ]] |

[[2 to 4 sentences: what the table shows. Expected shape: thresholds catch the LR and physics blow-ups but miss the reward bugs; Gemma catches the flips; the rules catch stall and crash for everyone. Only write what your table actually says.]]

Gemma on its own, per test episode: [[TP]] / [[FP]] / [[FN]] / [[TN]], precision [[ ]], recall [[ ]], median [[ ]] s per look on my laptop ([[CPU/GPU]]).

TabPFN on its own: AUROC [[ ]] over [[N]] windows, leave-one-run-out.

### What it can't see

- **The head camera answers one question well: which way up.** It doesn't see a robot that is upright but limping, crawling or walking backwards. Gemma couldn't judge those from the side view either, in my tests.
- **The camera is per robot.** Where to bolt it is one line per robot in `cameras.py` (HalfCheetah, Hopper, Walker2d, Ant, Humanoid have one); I only evaluated HalfCheetah.
- **My first camera caused a false alarm.** I put it on the robot's nose. The healthy run learned to crouch nose-down, the camera went through the floor, and Gemma faithfully reported the floor in a strange place. It now sits in the middle of the body and only draws the world, not the robot: an artificial horizon.
- **Is a head camera cheating?** If all you want to know is "is it upside down", log the torso angle; that's one line of code, and you should. The watcher is meant to work from what a training script already produces (metrics and pictures) and catch the things you didn't write a check for. The head camera is the cheapest picture a 4B model reads reliably; a spinning robot shows up in it too (the floor moves between snapshots).
- **Four snapshots aren't a video.** A gait that's slowly getting worse can slip past.
- **TabPFN has seen 12 runs of one robot.** A different robot or algorithm needs its own examples first (`nightwatch fit` on your own runs).
- **The crash and stall were caught by the plain rules, not by AI.** That's fine, and it's why the rules are there.
- **Ground truth is my definition.** "Flipped" means body tilt over 120 degrees for at least half of a test episode, two test episodes in a row. Another threshold would move the numbers.
- [[anything that went wrong for you: a false alarm and why, a missed failure and why]]

## Why Does Open Innovation Matter?

**The pictures stay home.** These are frames from unpublished research. A look every 5 minutes is about 100 images a night, every night, and I'd rather none of them leave my laptop.

**It runs where the compute is.** Lab and cluster machines often have no outbound internet. Ollama, TabPFN's local weights and a Temporal dev server all run offline.

**It costs nothing per night**, so I don't ration how often it looks.

**I could read and change every part.** I swapped Gemma sizes with one setting (`gemma4:e2b` on CPU, `gemma4:e4b` on [[your GPU]]), read TabPFN's source to choose a model version, and ran exactly the same models in the evaluation as in the live watch.

**Where a closed model would be better:** judging the side view. A frontier vision model would very likely see the flip from the side camera that my 4B model was blind to, and could watch a short video instead. But I only found out my model was blind because I could test it cheaply and repeatedly on my own laptop, and the fix (a camera) came from owning the whole stack, renderer included. A closed model would also mean ~400 images a night leaving my machine, and a bill.

## My Agent Session

[[Optional: link a DevRelay session, or delete this section]]

## The Lazy Payoff

[[Fill with what really happened this week. For example:]]
- nights watched: [[N]]; times it woke me: [[N]], of which real problems: [[N]]
- what I used to do: [[e.g. check twice a night, about 15 minutes each including falling back asleep]]
- the run it saved: [[e.g. the flip-bug run was stopped at 01:40 instead of found at 08:00, saving N GPU-hours]]

## Prize Categories

- **Best Use of TabPFN**: TabPFN-3.5 (local open weights, laptop CPU) is the metrics watcher. It learns in context from a table of past runs' log windows labelled healthy or broken, and is evaluated leave-one-run-out against simulator ground truth. The live watcher fits once with `fit_with_cache`, so each 5-minute check takes under a second instead of ~96 s.
- **Best Use of Gemma**: Gemma 4 E4B (QAT build, on a laptop iGPU through Ollama) looks through a head camera bolted to the robot and says where the floor is, in a fixed JSON schema; code turns that into upright / upside down / spinning. Its eyesight is measured against the simulator (`nightwatch eyetest`), including the side view it could not judge. It also writes the morning summary, whose numbers are checked against the facts.
- **Best Use of Temporal**: the overnight watch is a Temporal workflow with durable timers, retried activities for the flaky model calls, continue-as-new with carried state, stop/snooze signals and a status query. It survived the worker being killed mid-night.
- [[Best Use of Sentry Agent Tracing: only if you set SENTRY_DSN and include a trace screenshot of a Gemma call]]

Built solo.
