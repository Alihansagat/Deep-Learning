# Held-out test failure analysis

The selected baseline made 1,158 errors among 6,413 test crops. Visual review of three low-confidence errors:

| Image | Ground truth | Prediction | Confidence | Likely cause |
|---|---|---|---:|---|
| `20150708_1300_40.jpg` | busy | free | 0.502 | Dense foliage hides most of the vehicle and slot cues. |
| `20150708_1410_46.jpg` | busy | free | 0.503 | Partial car visibility combined with strong leaf shadows makes the crop ambiguous. |
| `20150708_1755_36.jpg` | free | busy | 0.504 | A neighboring vehicle wheel enters the crop although the target space is free. |

All three sit near the 0.50 decision threshold. The held-out camera/date shift is likely contributing as well. A practical next step is to add diverse, balanced examples from additional viewpoints, including foliage, shadows, and neighboring-car occlusion, then evaluate on a new site-level holdout.

The complete per-image predictions are in `test_predictions.csv`; structured details are in `failure_cases.json`.
