# Testing Night Watch on Windows, step by step

Total hands-on time is about 1.5 hours. The rest is waiting (training, Gemma looking at pictures, and a real night). Everything below runs in PowerShell from the repo folder.

## 0. One-time installs (15 min)

1. **Python 3.11 or 3.12, 64-bit** from python.org. Tick "Add python.exe to PATH".
2. **Git** (git-scm.com).
3. **Ollama for Windows** (ollama.com). It runs in the tray after install.
4. **Temporal CLI**: download https://temporal.download/cli/archive/latest?platform=windows&arch=amd64, unzip it, and put `temporal.exe` somewhere on your PATH (or just in the repo folder).
5. **Phone alerts (2 min)**: install the **ntfy** app on your phone and subscribe to a topic with a long random name, e.g. `nightwatch-suhas-7f3k9q`. Anyone who knows the name can read it, so make it unguessable.
6. **Stop Windows from sleeping while plugged in**: Settings > System > Power > Screen and sleep > "When plugged in, put my device to sleep after: Never". A sleeping laptop pauses training *and* the watcher. Temporal will resume the watch afterwards, but the run itself is paused.

## 1. Set up the repo (10 min)

```powershell
git clone https://github.com/su0as/night-watch.git
cd night-watch
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # only if activation is blocked
.\.venv\Scripts\Activate.ps1
# NVIDIA GPU? Install CUDA PyTorch first so TabPFN and Gemma-side work are faster:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
ollama pull gemma4:e4b          # about 10 GB; use gemma4:e2b on a weak laptop
copy .env.example .env
notepad .env                     # set NW_NTFY_TOPIC=<your topic>
```

## 2. Check every moving part (2 min)

```powershell
python -m nightwatch doctor
python -m nightwatch ping        # your phone should buzz
python -m pytest -q              # 21 pure-Python tests + the 8-hour Temporal night in seconds
```

`doctor` checks packages, that MuJoCo can render a HalfCheetah frame, that Ollama is up and Gemma is pulled, that TabPFN imports, and that Temporal is reachable (start it first if you want that line green: `temporal server start-dev --db-filename nightwatch.db`).

If the Temporal test is skipped, temporalio did not import: run `python -c "import temporalio"` to see why.

## 3. Smoke run: the whole pipeline on tiny runs (15 min)

```powershell
python -m nightwatch runs --smoke --jobs 3
```

This trains 4 tiny runs (60k steps each): healthy, the flipped-sign reward bug, a learning-rate blow-up at 20k, and a crash at 30k. The same smoke runs in CI on every push (GitHub Actions, Linux with software rendering and Windows without), so if it fails on your laptop, compare with the latest CI run. Then check:

- `runs\smoke\smoke-flip\frames\` has PNGs, and the `*_sheet.png` files show a robot (not black). **This matters: Gemma can only judge what it can see.**
- `runs\smoke\smoke-crash\manifest.json` still says `"status": "running"`. That is correct: a crash leaves no goodbye.

```powershell
python -m nightwatch fit  --runs runs\smoke --out models\smoke.csv
python -m nightwatch look --runs runs\smoke
python -m nightwatch eval --runs runs\smoke --out docs\EVAL_smoke.md --profiles thresholds vision
```

**TabPFN licence:** the first time TabPFN fits (during `eval` with the `tabpfn` profile, or `watch`/`worker`), it opens a browser to log in at Prior Labs and accept the non-commercial licence. If that does not work, get a token from https://ux.priorlabs.ai (Licence tab) and put `TABPFN_TOKEN=...` in `.env`.

Open `docs\EVAL_smoke.md`. Numbers on 3 tiny runs mean nothing; you are only checking that every section fills in.

## 4. The real dataset (about 1 hour unattended, plus one night for Gemma)

```powershell
python -m nightwatch runs --jobs 3      # 12 runs x 400k steps
```

On a laptop CPU each run takes roughly 10 to 25 minutes, 3 at a time. Logs are in `runs\dataset\*.log`. The stall run deliberately hangs for 15 minutes before it exits.

**Check the reward-bug runs actually flipped**, because this is the story of the post. Open the last `*_sheet.png` in `runs\dataset\flip-s1\frames\`. If the robot is not on its back, raise the bug weight and rerun just that one:

```powershell
Remove-Item -Recurse runs\dataset\flip-s1
python -m nightwatch train --fault "flip_bug:weight=2" --seed 1 --name flip-s1 --runs-dir runs\dataset
```

Then:

```powershell
python -m nightwatch fit  --runs runs\dataset            # -> models\windows.csv
python -m nightwatch look --runs runs\dataset            # Gemma: ~250 looks. Fine to leave overnight.
python -m nightwatch eval --runs runs\dataset            # -> docs\EVAL.md and docs\EVAL.json
```

`look` prints every verdict and caches it, so you can stop and restart it any time. `eval` itself takes minutes, because Gemma's answers are cached.

**Before quoting any TabPFN number, check `docs\EVAL.md` says the backend is `tabpfn-local-v3.5`** (or `tabpfn-client`), not `sklearn-fallback`.

## 5. A real night with Temporal (set up at bedtime: 5 min)

Open four PowerShell windows (all with the venv activated), or run `scripts\night.ps1`, which opens them for you:

```powershell
# 1
temporal server start-dev --db-filename nightwatch.db
# 2
python -m nightwatch worker
# 3  the run you actually care about. Try a buggy one on the first night so there is something to catch:
python -m nightwatch train --preset zoo --steps 3000000 --fault flip_bug --name night-1
# 4  once run 3 has written its first metrics (about a minute):
python -m nightwatch start runs\night-1 --until 07:30
```

Temporal UI: http://localhost:8233. You will see the `nightwatch-night-1` workflow with one check every 5 minutes.

### The kill test (the Temporal part of the post)

Around midnight, close window 2 (the worker). The training keeps going and the workflow stays "Running" on the server with nothing executing it. Take a screenshot. Wait 15 to 30 minutes, then start the worker again:

```powershell
python -m nightwatch worker
```

The workflow picks up from its last completed check, and it still remembers how many times it has woken you. Screenshot the event history in the UI around the gap.

For contrast, do the same thing with the plain loop (`python -m nightwatch watch runs\night-1 --until 07:30`) on another night: close it and the watch is simply over.

### In the morning

- `runs\night-1\report\morning.md`, `timelapse.gif`, `reward.png`
- your phone: the wake-up(s) and the morning summary
- `python -m nightwatch status runs\night-1` (if still running)

## 6. What to capture for the post

- [ ] `timelapse.gif` from a flip run: the robot ends up on its back
- [ ] `reward.png` from the same run: reward climbing while it flips, with the wake-up marked
- [ ] phone screenshot of the 3 a.m. wake-up with the contact sheet attached
- [ ] Temporal UI: the gap where the worker was dead, then the checks resuming
- [ ] `docs/EVAL.md` tables (the post quotes them)
- [ ] the Gemma confusion matrix and its median latency on your laptop
- [ ] (optional) a Sentry trace of one Gemma call: set `SENTRY_DSN` in `.env`
- [ ] a 60 to 90 second screen recording: start the watch, cut to the morning report and the phone alert

## Troubleshooting

| symptom | fix |
|---|---|
| frames are black or rendering fails | update the GPU driver; try `set MUJOCO_GL=glfw`; `doctor` shows the error |
| Gemma takes minutes per look | use `NW_VISION_MODEL=gemma4:e2b`; close other GPU apps; CPU-only is slow but fine for `look` overnight |
| `TabPFN ... CPU ... 5000 samples` | set `TABPFN_ALLOW_CPU_LARGE_DATASET=true`, or use `NW_TABPFN_BACKEND=client` |
| Windows crash `0xc000001d` with TabPFN | set `ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16` (known PyTorch/oneDNN issue) |
| `start` says the workflow already exists | the id is `nightwatch-<run name>`; use a new run name, or `stop` the old one |
| no phone alerts | `python -m nightwatch ping`; check the topic name in `.env` and in the ntfy app match |

If you change any code after 25 October 2026, 11:59 PM PDT, note it in the README (challenge rule).
