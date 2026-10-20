<!--
DRAFT for the DEV post. Do not publish with any [[...]] left in it.
- [[N]]-style brackets = a number you measured (from docs/EVAL.md, the morning report, or your phone).
- [[TRUE? ...]] = a personal detail; keep it only if it is true, otherwise rewrite it.
- Keep it under ~2,500 words. Keep the tone plain. Delete this comment block before publishing.

Title options (pick one after you see the results):
1. My robot's reward went up all night. It had learned to lie on its back.
2. I wanted to sleep while my robot learned to run, so two open models watch it for me
3. Night Watch: TabPFN reads my training logs, Gemma watches my robot, and I sleep
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

- **Gemma 4** (running locally through Ollama) looks at snapshots of the robot every 20k training steps and compares them with the robot's own first picture;
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

### Gemma 4 looks at the robot

Every 20k steps the trainer plays a short test episode and saves four snapshots as one 2x2 image. Gemma gets **two** images: the robot's very first sheet (untrained, but standing the right way up) as a reference, and the newest sheet. That one trick means there's no robot-specific code for "what upright looks like".

It answers in a fixed JSON schema: `posture` (upright, upside_down, on_its_side, mid_flip, broken, unclear) and `motion` (moving_forward, spinning, stuck...), plus a short "why". **Only the enum fields decide a wake-up.** The "why" goes into the message so I know what it saw, but a chatty answer can't wake me on its own. It also has to see the problem twice in a row.

[[One or two real examples from docs/EVAL.json vision_rows: what it said for a flipped sheet, and one it got wrong]]

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

- **Four snapshots aren't a video.** A gait that's slowly getting worse, or a robot that's upright but walking backwards, can slip past.
- **TabPFN has seen 12 runs of one robot.** A different robot or algorithm needs its own examples first (`nightwatch fit` on your own runs).
- **The crash and stall were caught by the plain rules, not by AI.** That's fine, and it's why the rules are there.
- **Ground truth is my definition.** "Flipped" means body tilt over 120 degrees for at least half of a test episode, two test episodes in a row. Another threshold would move the numbers.
- [[anything that went wrong for you: a false alarm and why, a missed failure and why]]

## Why Does Open Innovation Matter?

**The pictures stay home.** These are frames from unpublished research. A look every 5 minutes is about 100 images a night, every night, and I'd rather none of them leave my laptop.

**It runs where the compute is.** Lab and cluster machines often have no outbound internet. Ollama, TabPFN's local weights and a Temporal dev server all run offline.

**It costs nothing per night**, so I don't ration how often it looks.

**I could read and change every part.** I swapped Gemma sizes with one setting (`gemma4:e2b` on CPU, `gemma4:e4b` on [[your GPU]]), read TabPFN's source to choose a model version, and ran exactly the same models in the evaluation as in the live watch.

**Where a closed model would be better:** judging the pictures. A frontier vision model would probably tell "mid-flip" from "upside down" more reliably from four frames, and could watch a short video instead. It would also need the same checks around it, and it would mean a bill and a data flow I'd have to think about every single night.

## My Agent Session

[[Optional: link a DevRelay session, or delete this section]]

## The Lazy Payoff

[[Fill with what really happened this week. For example:]]
- nights watched: [[N]]; times it woke me: [[N]], of which real problems: [[N]]
- what I used to do: [[e.g. check twice a night, about 15 minutes each including falling back asleep]]
- the run it saved: [[e.g. the flip-bug run was stopped at 01:40 instead of found at 08:00, saving N GPU-hours]]

## Prize Categories

- **Best Use of TabPFN**: TabPFN-3.5 (local open weights) is the metrics watcher. It learns in context from a table of past runs' log windows labelled healthy or broken, and is evaluated leave-one-run-out against simulator ground truth.
- **Best Use of Gemma**: Gemma 4 E4B runs locally through Ollama. It looks at a reference picture and the latest snapshots of the robot, answers in a fixed JSON schema, and writes the morning summary, whose numbers are checked against the facts.
- **Best Use of Temporal**: the overnight watch is a Temporal workflow with durable timers, retried activities for the flaky model calls, continue-as-new with carried state, stop/snooze signals and a status query. It survived the worker being killed mid-night.
- [[Best Use of Sentry Agent Tracing: only if you set SENTRY_DSN and include a trace screenshot of a Gemma call]]

Built solo.
