# Running Night Watch on a Windows laptop

Tested end to end on a Lenovo ThinkPad (Intel i7-1165G7, 4 cores, 16 GB RAM, Iris Xe, no NVIDIA GPU, Windows 11), with everything installed on the D: drive. Timings below are from that laptop.

Hands-on time is about an hour. The rest is waiting (training, Gemma looking at pictures, one real night).

## 0. One-time installs

Normal installers work (Python 3.11/3.12 64-bit, Git, Ollama for Windows, Temporal CLI). If your C: drive is full, or you have no admin rights, everything also runs portable from another drive. This is what the test laptop uses, all under `D:\dev`:

| tool | portable version | where |
|---|---|---|
| Git | MinGit zip from git-scm.com | `D:\dev\dev-tools\git` |
| Python | `uv` + a standalone CPython 3.12 (`uv venv --python <path>`) | `D:\dev\dev-tools\uv`, `py312` |
| Ollama | `ollama-windows-amd64.zip` from the GitHub releases page (1.4 GB unzipped) | `D:\dev\dev-tools\ollama` |
| Temporal CLI | https://temporal.download/cli/archive/latest?platform=windows&arch=amd64 | `D:\dev\dev-tools\temporal` |

Keep the big downloads off C: too:

- `OLLAMA_MODELS=D:\dev\ollama-models` (Gemma is 6 GB)
- `HF_HOME`, `TORCH_HOME`, `TABPFN_MODEL_CACHE_DIR`, `UV_CACHE_DIR`, `TEMP`/`TMP` pointing at D:

A small env script that sets these, puts the tools on PATH and activates the venv saves retyping. Dot-source it in every new PowerShell window.

**Intel or AMD integrated graphics:** start Ollama with `OLLAMA_IGPU_ENABLE=1`. Ollama drops integrated GPUs by default; on the Iris Xe (Vulkan) prompt processing went from about 9 to 23 tokens/s, roughly twice as fast as the CPU.

**Phone alerts:** install the **ntfy** app and subscribe to a topic with a long random name (e.g. `nightwatch-yourname-7f3k9q2m`). Anyone who knows the name can read it.

## 1. Set up the repo

```powershell
git clone https://github.com/su0as/night-watch.git
cd night-watch
python -m venv .venv            # or: uv venv --python <path to python.exe> .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"          # CPU PyTorch is what pip installs on Windows; that is fine
ollama pull gemma4:e4b-it-qat    # 6.1 GB. The QAT build is the smallest E4B; plenty on a 16 GB laptop
copy .env.example .env
notepad .env                     # NW_NTFY_TOPIC, NW_VISION_MODEL=gemma4:e4b-it-qat, TABPFN_TOKEN (below)
```

**TabPFN licence on Windows: use a token.** TabPFN's first-run login opens a browser and then waits for you to paste a key using `select()` on the terminal, which Windows does not support for console input, so the prompt crashes. Instead:

1. Log in at https://ux.priorlabs.ai (GitHub login works), open **Licenses**, accept the TabPFN-3.5 non-commercial licence.
2. **API Keys** > Create Key, copy it.
3. Put `TABPFN_TOKEN=...` in `.env` (git-ignored). The TabPFN-3.5 weights (835 MB) then download on first use.

## 2. Check every moving part

```powershell
python -m nightwatch doctor
python -m nightwatch ping        # your phone should buzz
python -m pytest -q              # 31 tests, including an 8-hour Temporal night in a few seconds
```

`doctor` now also checks the things that ruin a night on a laptop:

- **Free RAM and commit.** Gemma E4B takes ~6 GB, TabPFN ~1.5 GB, each training run ~0.5 GB. On the test laptop, with a browser open, Windows was down to under 1 GB of commit (RAM + page file) and Ollama's model server died once mid-test. Close big apps at night. If your page file lives on a nearly full C:, add one on another drive (System Properties > Advanced > Performance > Virtual memory; needs admin and a reboot).
- **Sleep.** The test laptop was set to sleep after 10 minutes on AC. Training, `watch` and the Temporal worker now ask Windows to stay awake while they run (`SetThreadExecutionState`, the same call video players use), but set "When plugged in, put my device to sleep after: Never" anyway, and don't close the lid unless the lid action is "Do nothing".
- **TabPFN licence token** present.

## 3. Can Gemma actually see the robot? (20 minutes)

```powershell
python -m nightwatch eyetest
```

This drops the robot into known poses (right way up, on its back), lets the physics settle it, takes four snapshots like a real test episode, and asks the same Gemma judge that runs at night. Results go to `docs/EYETEST.md`. On the test laptop:

| what Gemma is shown | right calls | upside-down robots caught | seconds per test episode |
|---|---|---|---|
| side camera, 2x2 sheet of 320x240 snapshots (the original design, 12 episodes) | 6 / 12 | 0 / 6 | 10 |
| side camera, full-size 1024x768 snapshots | 12 / 18 | 0 / 6 | 22 |
| **head camera bolted to the robot, 512x384** | **18 / 18** | **6 / 6** | 25 |
| head camera, but the smaller `gemma4:e2b-it-qat` | 12 / 18 | 0 / 6 | 18 |

From the side, Gemma 4 E4B called every robot "upright". Bigger pictures, other wordings, a reference picture and thinking mode (6 minutes per look on this laptop) did not fix it, and the 12B model does not fit the iGPU. From a camera fixed in the robot's body the question becomes "where are the floor and the sky in this picture?", which it answers every time, including for a robot crouching nose-down (that case is in the test because it fooled the first camera mount). That is why the trainer now renders a head camera, and the vision watcher only asks Gemma where the floor and sky are. Plain code turns the four answers into a posture.

## 4. Smoke run: the whole pipeline on tiny runs (about 25 minutes)

```powershell
python -m nightwatch runs --smoke --jobs 3
python -m nightwatch fit  --runs runs\smoke --out models\smoke.csv
python -m nightwatch look --runs runs\smoke
python -m nightwatch eval --runs runs\smoke --out docs\EVAL_smoke.md
```

On the test laptop: 4 runs of 60k steps in about 5 minutes (3 at a time), `fit` in seconds, `look` about 25 s per test episode (23 episodes), `eval` under a minute. Result (`docs/EVAL_smoke.md`): the full watcher caught all 3 failures (flip, LR blow-up, crash; median 0.3 minutes to wake), and Gemma on its own was right on all 23 test episodes (6 flipped, 17 fine). TabPFN with only 3 training runs per fold is not meaningful yet (AUROC 0.42, one false wake); that is what the 12-run dataset is for.

`eval` unloads Gemma from Ollama before it loads TabPFN: on 16 GB with a browser open, TabPFN's 835 MB of weights would not load next to Gemma (Windows error 1455, or an access violation inside torch).

Check:

- `runs\smoke\smoke-flip\frames\` has `*_sheet.png` (side view, for you) and `*_head0..3.png` (head camera, for Gemma).
- `runs\smoke\smoke-crash\manifest.json` still says `"status": "running"`. That is correct: a crash leaves no goodbye.
- `docs\EVAL_smoke.md` says the TabPFN backend is `tabpfn-local-v3.5`, not `sklearn-fallback`.

Training runs use one torch thread each when three run at once (cores / jobs). With every run grabbing all four cores they fought each other: 485 vs 610 steps/s per run.

## 5. The real dataset (about 1.5 hours unattended, plus the Gemma pass)

```powershell
python -m nightwatch runs --jobs 3      # 12 runs x 400k steps
python -m nightwatch fit  --runs runs\dataset            # -> models\windows.csv
python -m nightwatch look --runs runs\dataset            # Gemma: ~250 test episodes, ~20 s each
python -m nightwatch eval --runs runs\dataset            # -> docs\EVAL.md and docs\EVAL.json
```

Logs are in `runs\dataset\*.log`. The stall run deliberately hangs for 15 minutes before it exits. `look` caches every verdict, so you can stop and restart it.

Note: even a run with no bug can learn to run on its back. HalfCheetah is known for finding that trick by itself (the healthy smoke run did, by step 60k). The evaluation counts it as a real failure, because it is one: the simulator says the robot is upside down, whatever the reward says.

## 6. A real night with Temporal (5 minutes at bedtime)

Four windows (each with the venv active), or `.\scripts\night.ps1 -Name night-1 -Fault flip_bug`, which opens them for you:

```powershell
temporal server start-dev --db-filename nightwatch.db           # 1
python -m nightwatch worker                                     # 2
python -m nightwatch train --preset zoo --steps 3000000 --fault flip_bug --name night-1   # 3
python -m nightwatch start runs\night-1 --until 07:30           # 4, once run 3 has logged metrics
```

The live watcher fits TabPFN once on your past runs and keeps its attention cache (`fit_with_cache`). Each 5-minute check then takes under a second on the laptop CPU, instead of about 96 s re-reading the whole table every time.

Tested on the laptop with a 600k-step flip-bug run, checks every 60 s (`--every 60`) and the worker killed after 5 minutes:

| time | what happened |
|---|---|
| 00:38 | woke: Gemma "upside down, floor/sky top/bottom in 4/4 snapshots" (step 100k), and TabPFN p=0.86 |
| 00:42:00 | worker killed |
| 00:42:15 | the 1-minute durable timer fired on the server anyway; the next workflow task timed out with no worker |
| 00:46:15 | worker restarted; it picked the workflow up and carried on from its last check |
| 00:46 | same problem seen again: a morning note, not a second wake (the wake count survived the kill) |
| 00:49:18 | training finished; morning report written by Gemma (numbers checked) and sent |

On 16 GB, set `NW_OLLAMA_KEEP_ALIVE=2m` so Ollama drops Gemma between checks, and start the worker before bed: it now fits TabPFN and checks every model at startup (`[worker] all detectors ready`), so a missing licence or a memory problem shows up while you are awake. If a model still fails at night, only its own check is lost (a morning note); the plain rules never depend on a model.

### The kill test (the Temporal part of the post)

Around midnight, close window 2 (the worker). Training keeps going, and the workflow stays "Running" on the server with nothing executing it. Screenshot http://localhost:8233. Wait 15 to 30 minutes, start the worker again, and the workflow carries on from its last completed check, still knowing how many times it has woken you. Screenshot the event history around the gap.

For contrast, do the same with the plain loop (`python -m nightwatch watch runs\night-1 --until 07:30`) on another night: close it and the watch is simply over.

## 7. What to capture for the post

- [ ] `docs/assets/eyetest_side_vs_head.png` and the `docs/EYETEST.md` table
- [ ] `timelapse.gif` and `reward.png` from a flip run
- [ ] phone screenshot of a wake-up with the picture attached
- [ ] Temporal UI: the gap where the worker was dead, then the checks resuming
- [ ] `docs/EVAL.md` tables
- [ ] a 60 to 90 second screen recording

## Troubleshooting

| symptom | fix |
|---|---|
| side frames half black | two OpenGL contexts fighting; fixed by rendering both cameras from one renderer (`nightwatch/cameras.py`). If you still see it, update the GPU driver |
| Gemma says the same posture for every picture | you are judging the side view (old runs, or `NW_VISION_VIEW=side`). Re-record with the head camera |
| Gemma looks are slow | start Ollama with `OLLAMA_IGPU_ENABLE=1`; close other GPU apps. Don't drop to `gemma4:e2b`: in the eyetest it caught 0 of 6 upside-down robots even with the head camera (E4B: 6 of 6) |
| `ErrorOutOfDeviceMemory` from Ollama | the model is too big for the iGPU's share of RAM (gemma4:12b on 16 GB). Use E4B |
| Ollama stops overnight | it was started from a terminal that closed, or Windows ran out of commit. Start it detached and leave headroom |
| TabPFN login prompt crashes | Windows console `select()` limitation; set `TABPFN_TOKEN` in `.env` |
| `TabPFN ... CPU ... samples` warning | harmless at this size; `TABPFN_ALLOW_CPU_LARGE_DATASET=true` silences it |
| `start` says the workflow already exists | the id is `nightwatch-<run name>`; use a new run name, or `stop` the old one |
| no phone alerts | `python -m nightwatch ping`; check the topic name in `.env` and in the ntfy app match |

If you change any code after 25 October 2026, 11:59 PM PDT, note it in the README (challenge rule).
