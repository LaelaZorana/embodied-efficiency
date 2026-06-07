"""Build a labelled OOD / fault evaluation set for the safety supervisor.

The supervisor in this repo is calibrated on whatever "normal" looks like and
then flags actions that drift away from it. To say how *well* it does that we
need real in-distribution actions plus labelled faults, so this script:

  1. pulls a small slice of real teleoperated actions from DROID (a Franka arm,
     7-dim action [x, y, z, roll, pitch, yaw, gripper]), via LeRobot on the Hub,
  2. splits episodes into a calibration set and a held-out test set,
  3. derives per-dimension action limits from the calibration set,
  4. builds a balanced eval set: real actions (label 0, "normal") plus four
     realistic fault types (label 1) injected into held-out real actions:
        - nonfinite     a sensor glitch puts NaN/Inf in one dimension
        - out_of_bounds one or more dims pushed past the joint limits
        - drift         the action shifted k std away from the calibrated posture
        - jerk          a large in-bounds jump from the previous accepted action

The in-distribution data is real; the faults are synthetic but injected into
real actions and labelled as such (honest by construction). Output is a single
self-contained .npz (small) plus a data card, committed so evaluate.py needs
only numpy.

Run:  python3 safety/make_eval_set.py        # writes safety/data/supervisor_eval.npz
Deps (build-time only): huggingface_hub, pyarrow.  Consuming it needs only numpy.

Source dataset: IPEC-COMMUNITY/droid_lerobot (DROID, CC-BY-4.0). DROID:
A Large-Scale In-the-Wild Robot Manipulation Dataset (Khazatsky et al., 2024).
"""
from __future__ import annotations

import glob
import os

import numpy as np

REPO = "IPEC-COMMUNITY/droid_lerobot"
N_EPISODES = 100          # episodes 0..99 from chunk-000 (low-dim parquet only)
N_CALIB_EP = 70           # first N_CALIB_EP episodes calibrate; the rest are held out for test
SEED = 0
OUT = os.path.join(os.path.dirname(__file__), "data", "supervisor_eval.npz")

FAULTS = ["nonfinite", "out_of_bounds", "drift", "jerk"]
FTYPE_CODE = {"normal": 0, "nonfinite": 1, "out_of_bounds": 2, "drift": 3, "jerk": 4}


def _download_actions():
    """Return a list of [T, 7] action arrays, one per episode, real DROID data."""
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq

    local = snapshot_download(
        REPO, repo_type="dataset",
        allow_patterns=["data/chunk-000/episode_0000*.parquet"],   # episodes 0..99
    )
    files = sorted(glob.glob(os.path.join(local, "data", "chunk-000", "*.parquet")))[:N_EPISODES]
    if not files:
        raise RuntimeError("no episode parquets downloaded; check network / dataset id")
    eps = []
    for f in files:
        col = pq.read_table(f, columns=["action"]).column("action").to_pylist()
        eps.append(np.asarray(col, dtype=np.float64))   # [T, 7]
    return eps


def main():
    rng = np.random.default_rng(SEED)
    eps = _download_actions()
    calib_eps, test_eps = eps[:N_CALIB_EP], eps[N_CALIB_EP:]

    calib = np.concatenate(calib_eps, axis=0)
    A = calib.shape[1]
    mean, std = calib.mean(0), calib.std(0)

    # Joint limits from the calibration range, with a 15% margin so legitimate
    # held-out actions stay in-bounds and only genuine over-range trips the check.
    lo, hi = calib.min(0), calib.max(0)
    margin = 0.15 * (hi - lo)
    low, high = lo - margin, hi + margin

    # subsample calibration so the committed file stays small
    if len(calib) > 6000:
        calib = calib[rng.choice(len(calib), 6000, replace=False)]

    # Held-out real (prev, action) consecutive pairs: the source of every eval row.
    pairs = []
    for ep in test_eps:
        for t in range(1, len(ep)):
            pairs.append((ep[t - 1], ep[t]))
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]

    per = 600                                   # rows per class -> ~3000 total, balanced 50/50
    need = per * (1 + len(FAULTS))
    pairs = pairs[:need]
    chunks = {k: pairs[i * per:(i + 1) * per]
              for i, k in enumerate(["normal"] + FAULTS)}

    ev_act, ev_prev, ev_label, ev_ftype = [], [], [], []

    def add(action, prev, label, ftype):
        ev_act.append(action); ev_prev.append(prev)
        ev_label.append(label); ev_ftype.append(FTYPE_CODE[ftype])

    for prev, a in chunks["normal"]:
        add(a.copy(), prev.copy(), 0, "normal")

    for prev, a in chunks["nonfinite"]:
        f = a.copy(); f[rng.integers(A)] = np.nan if rng.random() < 0.5 else np.inf
        add(f, prev.copy(), 1, "nonfinite")

    for prev, a in chunks["out_of_bounds"]:
        f = a.copy()
        for d in rng.choice(A, rng.integers(1, 4), replace=False):
            over = (0.2 + rng.random()) * (high[d] - low[d])     # clearly past the limit
            f[d] = (high[d] + over) if rng.random() < 0.5 else (low[d] - over)
        add(f, prev.copy(), 1, "out_of_bounds")

    for prev, a in chunks["drift"]:
        k = rng.uniform(1.5, 6.0)                                # severity in std units, mild..strong
        direction = rng.normal(size=A); direction /= np.linalg.norm(direction) + 1e-9
        f = np.clip(a + k * std * direction * np.sqrt(A), low, high)   # drifted, kept in-bounds
        add(f, prev.copy(), 1, "drift")

    for prev, a in chunks["jerk"]:
        # large jump from prev, kept in-bounds and near the calibrated posture so
        # it isolates the jerk check rather than tripping bounds/drift.
        direction = rng.normal(size=A); direction /= np.linalg.norm(direction) + 1e-9
        f = np.clip(prev + rng.uniform(0.5, 0.9) * (high - low) * direction, low, high)
        add(f, prev.copy(), 1, "jerk")

    ev_act = np.asarray(ev_act); ev_prev = np.asarray(ev_prev)
    ev_label = np.asarray(ev_label, dtype=np.int64); ev_ftype = np.asarray(ev_ftype, dtype=np.int64)
    perm = rng.permutation(len(ev_act))         # shuffle so classes interleave

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        calib_actions=calib.astype(np.float32),
        action_low=low.astype(np.float32), action_high=high.astype(np.float32),
        eval_action=ev_act[perm].astype(np.float32), eval_prev=ev_prev[perm].astype(np.float32),
        eval_label=ev_label[perm], eval_ftype=ev_ftype[perm],
        ftype_codes=np.asarray([f"{k}={v}" for k, v in FTYPE_CODE.items()]),
    )
    n = len(ev_act)
    print(f"wrote {OUT}")
    print(f"  calibration frames: {len(calib)} (from {N_CALIB_EP} episodes), action_dim={A}")
    print(f"  eval rows: {n}  ({(ev_label == 0).sum()} normal, {(ev_label == 1).sum()} fault)")
    for k, c in FTYPE_CODE.items():
        cnt = int((ev_ftype == c).sum())
        if cnt:
            print(f"    {k:<14} {cnt}")
    print(f"  action limits: low={np.round(low,3).tolist()}")
    print(f"                 high={np.round(high,3).tolist()}")


if __name__ == "__main__":
    main()
