# Night Watch evaluation

Generated 2026-10-25 14:06 from 12 recorded training runs in `runs/clean`. Ground truth comes from the simulator's body state and the injected-fault log in `labels/`, which no watcher reads. Every watcher includes the same plain rules (NaN, stalled log, crash).

## Did it wake me when it should have?

| watcher | failures caught | median minutes to wake | false wake-ups | false wake-ups per 8 h of training |
|---|---|---|---|---|
| thresholds | 6 / 8 | 6.3 | 5 | 4.2 |
| tabpfn | 6 / 8 | 7.8 | 1 | 0.84 |

### By kind of failure

| watcher | robot looks wrong (reward bugs) caught | median min | training faults caught | median min | false wake-ups on healthy runs | false wake-ups before a failure |
|---|---|---|---|---|---|---|
| thresholds | 2 / 4 | 75.1 | 4 / 4 | 6.0 | 2 (on 4 runs) | 3 |
| tabpfn | 2 / 4 | 69.8 | 4 / 4 | 6.1 | 0 (on 4 runs) | 1 |

## Run by run

| run | what went wrong (ground truth) | thresholds | tabpfn |
|---|---|---|---|
| crash-a | crash at step 250,000 (injected) | woke after 12.0 min (rules/stalled), 1 false | woke after 12.0 min (rules/stalled) |
| flip-a | flipped at step 140,000 (behaviour) | **missed** | woke after 136.0 min (tabpfn/metrics_anomaly), 1 false |
| flip-b | flipped at step 60,000 (behaviour) | woke after 149.6 min (thresholds/kl_spike) | woke after 3.5 min (tabpfn/metrics_anomaly) |
| healthy-a | nothing | 1 false wake(s) | quiet |
| healthy-b | nothing | quiet | quiet |
| healthy-c | nothing | 1 false wake(s) | quiet |
| healthy-d | nothing | quiet | quiet |
| lr-a | lr_blowup at step 200,000 (injected) | woke after 0.1 min (thresholds/value_loss_spike) | woke after 0.1 min (tabpfn/metrics_anomaly) |
| lr-b | lr_blowup at step 150,000 (injected) | woke after 0.1 min (thresholds/kl_spike) | woke after 0.2 min (tabpfn/metrics_anomaly) |
| physics-a | flipped at step 180,000 (behaviour) | woke after 0.7 min (thresholds/value_loss_spike), 1 false | **missed** |
| spin-a | flipped at step 300,000 (behaviour) | **missed** | **missed** |
| stall-a | stall at step 250,000 (injected) | woke after 12.0 min (rules/stalled), 1 false | woke after 12.0 min (rules/stalled) |

## TabPFN on its own (leave one run out)

- backend: `tabpfn-local-v3.5`
- windows scored: 2144 (676 after something broke)
- AUROC: 0.93
- features per window: 22; total time 1119.2 s
- for comparison, the training step alone ("how far into training are we"): AUROC 0.793
- TabPFN without the training-progress feature: AUROC 0.932 (21 features)

Settings: check every 120 s in replay, tail 60 min after a run goes silent, max 3 wake-ups per run, snooze 60 min per kind.
