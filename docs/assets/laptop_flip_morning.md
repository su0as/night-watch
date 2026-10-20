# Night Watch: killtest

The killtest run in HalfCheetah-v5 finished with 598016 steps done. The reward increased from 908 to 3525, matching the best reward. A problem was noted at 00:38 where the robot looked upside down in the last two test runs.

_Summary written by: gemma_

![time-lapse of the night](timelapse.gif)

![reward overnight](reward.png)

| | |
|---|---|
| env | HalfCheetah-v5 |
| status | finished |
| steps start | 2048 |
| steps end | 600064 |
| reward start | 908 |
| reward end | 3525 |
| reward best | 3525 |
| test runs | 32 |
| wakes | 2 |
| notes | 3 |
| suppressed | 2 |
| watch from | 00:36 |
| watch to | 00:49 |

## Woke you

- **00:38** step 100,000 [vision/robot_looks_wrong] The robot looks wrong in the last 2 test runs: posture=upside_down, motion=unclear. Gemma: "head camera, floor/sky: top/bottom, top/bottom, top/bottom, top/bottom"
  ![robot_looks_wrong](../frames/s000100000_sheet.png)
- **00:38** step 106,496 [tabpfn/metrics_anomaly] Training metrics look like past runs that broke (p=0.86). Furthest from healthy: ent_last (+7.9 sd), std_last (-6.8 sd).

## For the morning

- 00:40 step 182,272 [tabpfn/metrics_anomaly] Metrics drifting toward past failures (p=0.77). Furthest from healthy: ent_last (+16.3 sd), std_last (-11.5 sd).
- 00:46 step 500,000 [vision/robot_looks_wrong] The robot looks wrong in the last 2 test runs: posture=upside_down, motion=unclear. Gemma: "head camera, floor/sky: top/bottom, top/bottom, top/bottom, top/bottom"
- 00:48 step 600,064 [rules/finished] Training finished at step 600,064.

_Watcher profile: nightwatch. Generated 2026-10-21 00:49._