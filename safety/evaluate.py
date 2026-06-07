"""Evaluate the safety supervisor on the labelled eval set, with real numbers.

Calibrates the supervisor on the real DROID calibration actions, then:

  - sweeps the drift threshold over the normal-vs-drift rows to get an ROC and
    AUC, and picks a data-driven operating point at a target false-positive
    rate, so the threshold is justified instead of a magic 4.0;
  - replays every eval row through the real supervisor (the genuine step() code
    path) and reports precision / recall / false-positive rate at both the
    default and the chosen threshold, plus the catch rate per fault type.

Needs only numpy + the eval set built by make_eval_set.py.
Run:  python3 safety/evaluate.py   (add --json safety/data/eval_report.json to save)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from supervisor import Supervisor, SupervisorConfig

DATA = os.path.join(os.path.dirname(__file__), "data", "supervisor_eval.npz")
TARGET_FPR = 0.01           # operating point: highest detection at <= 1% false positives
CODE_NAME = {0: "normal", 1: "nonfinite", 2: "out_of_bounds", 3: "drift", 4: "jerk"}


def roc(scores_neg, scores_pos):
    """ROC + AUC for a score where higher = more anomalous. Returns (fpr, tpr, thr, auc)."""
    thr = np.unique(np.concatenate([scores_neg, scores_pos]))
    thr = np.concatenate([[-np.inf], thr, [np.inf]])
    tpr = np.array([(scores_pos >= t).mean() for t in thr])
    fpr = np.array([(scores_neg >= t).mean() for t in thr])
    order = np.argsort(fpr)
    fo, to = fpr[order], tpr[order]
    auc = float(np.sum((fo[1:] - fo[:-1]) * (to[1:] + to[:-1]) / 2.0))   # trapezoid, version-agnostic
    return fpr, tpr, thr, auc


def confusion(pred_fault, label):
    tp = int(((pred_fault == 1) & (label == 1)).sum())
    fp = int(((pred_fault == 1) & (label == 0)).sum())
    tn = int(((pred_fault == 0) & (label == 0)).sum())
    fn = int(((pred_fault == 0) & (label == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, fpr=fpr, f1=f1)


def replay(sup, acts, prevs, drift_thresh):
    """Run every row through the real step() at a given drift threshold; return predicted-fault mask."""
    sup.cfg.drift_thresh = drift_thresh
    pred = np.zeros(len(acts), dtype=np.int64)
    for i in range(len(acts)):
        sup._last_safe = prevs[i].astype(np.float64).copy()      # control history for the jerk check
        _, iv = sup.step(acts[i])
        pred[i] = 0 if iv is None else 1
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="optional path to save the report as JSON")
    args = ap.parse_args()

    d = np.load(DATA, allow_pickle=False)
    calib = d["calib_actions"].astype(np.float64)
    low, high = d["action_low"].astype(np.float64), d["action_high"].astype(np.float64)
    acts, prevs = d["eval_action"].astype(np.float64), d["eval_prev"].astype(np.float64)
    label, ftype = d["eval_label"], d["eval_ftype"]
    A = calib.shape[1]

    cfg = SupervisorConfig(action_low=low, action_high=high)
    sup = Supervisor(cfg).calibrate(calib)

    # ---- ROC for the drift (OOD) detector: normal vs drift rows ----
    drift_score = np.array([sup.drift_score(a) for a in acts])
    is_norm, is_drift = (ftype == 0), (ftype == 3)
    fpr, tpr, thr, auc = roc(drift_score[is_norm], drift_score[is_drift])

    # data-driven operating point: lowest threshold with FPR <= target (max recall under the cap)
    ok = np.where(fpr <= TARGET_FPR)[0]
    op_thr = float(thr[ok[np.argmax(tpr[ok])]]) if len(ok) else float(thr[-1])
    op_tpr = float(tpr[thr == op_thr][0]); op_fpr = float(fpr[thr == op_thr][0])

    # ---- overall verdict at default (4.0) and the chosen threshold ----
    res = {}
    for name, t in [("default(4.0)", 4.0), (f"tuned({op_thr:.2f})", op_thr)]:
        pred = replay(sup, acts, prevs, t)
        c = confusion(pred, label)
        per_type = {CODE_NAME[ct]: round(float(pred[ftype == ct].mean()), 4)
                    for ct in sorted(set(ftype.tolist()))}
        res[name] = {"threshold": t, **c, "catch_rate_by_type": per_type}

    # ---- report ----
    print("=" * 64)
    print("Safety supervisor — evaluation on real DROID actions + labelled faults")
    print("=" * 64)
    print(f"calibration frames : {len(calib)}   action_dim : {A}")
    print(f"eval rows          : {len(acts)}  ({int(is_norm.sum())} normal, {int((label==1).sum())} fault)")
    print()
    print(f"Drift (OOD) detector ROC, normal vs drift:  AUC = {auc:.3f}")
    print(f"  operating point at <= {TARGET_FPR*100:.0f}% false positives:")
    print(f"    threshold {op_thr:.2f}  ->  detection {op_tpr*100:.1f}%  at  FPR {op_fpr*100:.2f}%")
    print(f"  (shipped default threshold is 4.0)")
    print()
    for name, r in res.items():
        print(f"All faults, threshold = {name}")
        print(f"  precision {r['precision']*100:5.1f}%   recall {r['recall']*100:5.1f}%   "
              f"FPR {r['fpr']*100:4.1f}%   F1 {r['f1']:.3f}")
        print("  catch rate by type: " +
              "  ".join(f"{k} {v*100:.0f}%" for k, v in r["catch_rate_by_type"].items()))
        print()

    if args.json:
        out = {"calibration_frames": len(calib), "action_dim": A, "eval_rows": len(acts),
               "drift_auc": round(auc, 4),
               "operating_point": {"target_fpr": TARGET_FPR, "threshold": round(op_thr, 4),
                                   "detection": round(op_tpr, 4), "fpr": round(op_fpr, 4)},
               "at_thresholds": res}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
