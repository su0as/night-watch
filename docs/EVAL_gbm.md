# Night Watch evaluation

Gradient-boosting baseline: same 12 runs, same 22 features and same leave-one-run-out split as docs/EVAL.md, run with NW_TABPFN_BACKEND=sklearn (scikit-learn HistGradientBoostingClassifier, default settings, 200 rounds, not tuned). The watcher is labelled 'tabpfn' in the tables below only because the eval names watchers by profile. These are NOT TabPFN numbers.

Generated 2026-10-25 19:31 from 12 recorded training runs in `runs\clean`. Ground truth comes from the simulator's body state and the injected-fault log in `labels/`, which no watcher reads. Every watcher includes the same plain rules (NaN, stalled log, crash).

## Did it wake me when it should have?

| watcher | failures caught | median minutes to wake | false wake-ups | false wake-ups per 8 h of training |
|---|---|---|---|---|
| gradient boosting (sklearn) | 6 / 8 | 7.5 | 4 | 3.36 |

### By kind of failure

| watcher | robot looks wrong (reward bugs) caught | median min | training faults caught | median min | false wake-ups on healthy runs | false wake-ups before a failure |
|---|---|---|---|---|---|---|
| gradient boosting (sklearn) | 2 / 4 | 69.5 | 4 / 4 | 7.0 | 1 (on 4 runs) | 3 |

## Run by run

| run | what went wrong (ground truth) | gradient boosting (sklearn) |
|---|---|---|
| crash-a | crash at step 250,000 (injected) | woke after 12.0 min (rules/stalled) |
| flip-a | flipped at step 140,000 (behaviour) | woke after 136.0 min (tabpfn/metrics_anomaly), 1 false |
| flip-b | flipped at step 60,000 (behaviour) | woke after 2.9 min (tabpfn/metrics_anomaly) |
| healthy-a | nothing | quiet |
| healthy-b | nothing | quiet |
| healthy-c | nothing | quiet |
| healthy-d | nothing | 1 false wake(s) |
| lr-a | lr_blowup at step 200,000 (injected) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| lr-b | lr_blowup at step 150,000 (injected) | woke after 1.9 min (tabpfn/metrics_anomaly) |
| physics-a | flipped at step 180,000 (behaviour) | **missed** |
| spin-a | flipped at step 300,000 (behaviour) | **missed**, 1 false |
| stall-a | stall at step 250,000 (injected) | woke after 12.0 min (rules/stalled), 1 false |

## TabPFN on its own (leave one run out)

- backend: `sklearn-fallback`
- windows scored: 2144 (676 after something broke)
- AUROC: 0.888
- features per window: 22; total time 9.0 s
- for comparison, the training step alone ("how far into training are we"): AUROC 0.793
- TabPFN without the training-progress feature: AUROC 0.89 (21 features)
- **Warning: this used the sklearn fallback, not TabPFN. Do not quote these numbers as TabPFN.**

Settings: check every 120 s in replay, tail 60 min after a run goes silent, max 3 wake-ups per run, snooze 60 min per kind.
