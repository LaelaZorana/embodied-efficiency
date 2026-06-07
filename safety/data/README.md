# Supervisor evaluation set

`supervisor_eval.npz` is a small labelled set for measuring the runtime safety
supervisor: real robot actions for the in-distribution ("normal") class, plus
realistic faults injected into held-out real actions for the anomaly class. It's
what turns "the supervisor catches NaN" into a measured detection rate and a
false-positive rate.

## How it's built

`python3 safety/make_eval_set.py` regenerates this file. It pulls a small slice
of **real teleoperated actions from DROID** (a Franka arm, 7-dim action
`[x, y, z, roll, pitch, yaw, gripper]`) via LeRobot on the Hub, splits episodes
into a calibration set and a held-out test set, derives per-dimension joint
limits from calibration, then builds a balanced eval set:

| class | label | how it's made |
|---|---|---|
| normal | 0 | real held-out actions, untouched |
| nonfinite | 1 | a real action with NaN/Inf in one dimension |
| out_of_bounds | 1 | a real action with 1–3 dims pushed past the joint limits |
| drift | 1 | a real action shifted 1.5–6 std from the calibrated posture (kept in-bounds) |
| jerk | 1 | a large in-bounds jump from the previous accepted action |

The in-distribution data is real; the faults are synthetic but injected into
real actions and labelled as such. Honest by construction: nothing synthetic is
presented as real.

## Contents (`np.load`)

- `calib_actions` `[6000, 7]` — real actions to calibrate the supervisor on
- `action_low`, `action_high` `[7]` — derived joint limits
- `eval_action` `[3000, 7]`, `eval_prev` `[3000, 7]` — each eval action and the action before it (for the jerk check)
- `eval_label` `[3000]` — 0 normal, 1 fault
- `eval_ftype` `[3000]` — fault-type code; `ftype_codes` holds the mapping

## Evaluate

`python3 safety/evaluate.py` calibrates the supervisor on `calib_actions`, builds
an ROC for the drift detector, picks a data-driven operating point, and reports
precision / recall / false-positive rate and the catch rate per fault type.

## Also on the Hub

Published as a standalone, viewer-browsable dataset (parquet + this bundle):
[LaelaZ/vla-action-anomaly-eval](https://huggingface.co/datasets/LaelaZ/vla-action-anomaly-eval).

## Source & license

Derived from **DROID** (`IPEC-COMMUNITY/droid_lerobot`, the LeRobot port),
released under **CC-BY-4.0**. DROID: *A Large-Scale In-the-Wild Robot
Manipulation Dataset*, Khazatsky et al., 2024 — <https://droid-dataset.github.io>.
This file redistributes a small slice of those actions plus synthetic faults,
under the same CC-BY-4.0, with attribution.
