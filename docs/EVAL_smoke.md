# Night Watch evaluation

Generated 2026-10-21 00:35 from 4 recorded training runs in `runs\smoke`. Ground truth comes from the simulator's body state and the injected-fault log in `labels/`, which no watcher reads. Every watcher includes the same plain rules (NaN, stalled log, crash).

## Did it wake me when it should have?

| watcher | failures caught | median minutes to wake | false wake-ups | false wake-ups per 8 h of training |
|---|---|---|---|---|
| thresholds | 2 / 3 | 6.0 | 0 | 0.0 |
| tabpfn | 1 / 3 | 12.0 | 1 | 72.73 |
| vision | 2 / 3 | 6.2 | 0 | 0.0 |
| nightwatch | 3 / 3 | 0.3 | 1 | 72.73 |

## Run by run

| run | what went wrong (ground truth) | thresholds | tabpfn | vision | nightwatch |
|---|---|---|---|---|---|
| smoke-crash | crash at step 30,000 (injected) | woke after 12.0 min (rules/stalled) | woke after 12.0 min (rules/stalled) | woke after 12.0 min (rules/stalled) | woke after 12.0 min (rules/stalled) |
| smoke-flip | flipped at step 40,000 (behaviour) | **missed** | **missed** | woke after 0.3 min (vision/robot_looks_wrong) | woke after 0.3 min (vision/robot_looks_wrong) |
| smoke-healthy | nothing | quiet | 1 false wake(s) | quiet | 1 false wake(s) |
| smoke-lr | lr_blowup at step 20,000 (injected) | woke after 0.1 min (thresholds/kl_spike) | **missed** | **missed** | woke after 0.1 min (thresholds/kl_spike) |

## TabPFN on its own (leave one run out)

- backend: `tabpfn-local-v3.5`
- windows scored: 84 (32 after something broke)
- AUROC: 0.418
- features per window: 22; total time 12.1 s

## Gemma on its own (does the robot look wrong?)

| | simulator says wrong | simulator says fine |
|---|---|---|
| Gemma says wrong | 6 | 0 |
| Gemma says fine | 0 | 17 |

- precision 1.0, recall 1.0, unjudged sheets 0
- model `gemma4:e4b-it-qat`, head camera: one 'where are the floor and the sky?' answer per snapshot, posture decided by code; median latency 25.69 s, median tokens per look 868

Settings: check every 120 s in replay, tail 60 min after a run goes silent, max 3 wake-ups per run, snooze 60 min per kind.
