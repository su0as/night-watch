# Night Watch evaluation

Generated 2026-10-21 03:22 from 12 recorded training runs in `runs\dataset`. Ground truth comes from the simulator's body state and the injected-fault log in `labels/`, which no watcher reads. Every watcher includes the same plain rules (NaN, stalled log, crash).

## Did it wake me when it should have?

| watcher | failures caught | median minutes to wake | false wake-ups | false wake-ups per 8 h of training |
|---|---|---|---|---|
| thresholds | 6 / 11 | 4.2 | 0 | 0.0 |
| tabpfn | 10 / 11 | 0.1 | 1 | 5.26 |
| vision | 11 / 11 | 0.4 | 0 | 0.0 |
| nightwatch | 11 / 11 | 0.1 | 1 | 5.26 |

## Run by run

| run | what went wrong (ground truth) | thresholds | tabpfn | vision | nightwatch |
|---|---|---|---|---|---|
| crash-s1 | flipped at step 120,000 (behaviour) | woke after 14.6 min (rules/stalled) | woke after 0.1 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| flip-s1 | flipped at step 40,000 (behaviour) | **missed** | woke after 0.2 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.2 min (tabpfn/metrics_anomaly) |
| flip-s2 | flipped at step 40,000 (behaviour) | **missed** | woke after 0.1 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| healthy-s1 | flipped at step 120,000 (behaviour) | **missed** | woke after 0.0 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.0 min (tabpfn/metrics_anomaly) |
| healthy-s2 | flipped at step 40,000 (behaviour) | **missed** | woke after 0.1 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| healthy-s3 | nothing | quiet | 1 false wake(s) | quiet | 1 false wake(s) |
| healthy-s4 | flipped at step 80,000 (behaviour) | woke after 5.9 min (thresholds/kl_spike) | woke after 1.0 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.4 min (vision/robot_looks_wrong) |
| lr-s1 | flipped at step 120,000 (behaviour) | woke after 1.7 min (thresholds/kl_spike) | woke after 0.1 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| lr-s2 | flipped at step 40,000 (behaviour) | woke after 2.3 min (thresholds/kl_spike) | woke after 0.2 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.2 min (tabpfn/metrics_anomaly) |
| physics-s1 | flipped at step 120,000 (behaviour) | woke after 2.5 min (thresholds/ev_collapse) | woke after 0.1 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| spin-s1 | flipped at step 80,000 (behaviour) | **missed** | **missed** | woke after 0.0 min (vision/robot_looks_wrong) | woke after 0.0 min (vision/robot_looks_wrong) |
| stall-s1 | flipped at step 120,000 (behaviour) | woke after 14.6 min (rules/stalled) | woke after 0.1 min (tabpfn/metrics_anomaly) | woke after 0.4 min (vision/robot_looks_wrong) | woke after 0.1 min (tabpfn/metrics_anomaly) |

## TabPFN on its own (leave one run out)

- backend: `tabpfn-local-v3.5`
- windows scored: 2144 (1564 after something broke)
- AUROC: 0.879
- features per window: 22; total time 764.9 s

## Gemma on its own (does the robot look wrong?)

| | simulator says wrong | simulator says fine |
|---|---|---|
| Gemma says wrong | 125 | 18 |
| Gemma says fine | 2 | 89 |

- precision 0.874, recall 0.984, unjudged sheets 0
- model `gemma4:e4b-it-qat`, head camera: one 'where are the floor and the sky?' answer per snapshot, posture decided by code; median latency 22.13 s, median tokens per look 872

Settings: check every 120 s in replay, tail 60 min after a run goes silent, max 3 wake-ups per run, snooze 60 min per kind.
