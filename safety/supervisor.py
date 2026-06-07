"""
Runtime safety supervisor for an embodied or agentic policy.

A policy that's fast isn't the same as a policy you can leave running on its own.
This sits between the policy and the actuator, watches every action it proposes,
and when an action drifts somewhere the policy was never calibrated for, it swaps
in a safe fallback and writes down why. That log is the governance trail: an
on-call engineer or a regulator can read exactly when the policy got overridden
and what tripped it.

What it checks, on every action:
  - shape: a malformed action is treated as unsafe, never crashes the loop.
  - finite: no NaN or Inf ever reaches a motor.
  - in-bounds: every dimension stays inside the action limits.
  - drift (OOD): how far the action sits from the calibration set, as a per-dim
    z-score pooled into one distance. This is a deliberately simple v0 (diagonal
    Gaussian); it catches gross drift, not subtle correlated shifts.
  - jerk: how big the jump is from the last accepted action, against the
    calibration jerk.

When a check trips, the supervisor returns a safe action (hold the last accepted
one, clipped to limits) and appends an intervention record. The log is capped, and
the running counts are kept separately so they stay exact even after trimming, so
this is safe to leave running. No GPU. It's the trust layer that rides on top of
the efficient policy: efficiency gets the model onto the robot, this is what lets
it stay there.
"""
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SupervisorConfig:
    action_low: np.ndarray          # [A] lower limit per dimension
    action_high: np.ndarray         # [A] upper limit per dimension
    drift_thresh: float = 4.0       # pooled z-distance that counts as out-of-distribution
    jerk_thresh: float = 4.0        # pooled z-distance on the step-to-step change
    eps: float = 1e-6
    max_log: int = 2000             # cap the kept records so the log can't grow without bound


@dataclass
class Intervention:
    step: int
    reasons: list
    drift: float
    jerk: float
    action_in: list
    action_out: list


@dataclass
class Supervisor:
    cfg: SupervisorConfig
    _t: int = 0
    _last_safe: np.ndarray = None
    _mean: np.ndarray = None
    _std: np.ndarray = None
    _jmean: np.ndarray = None
    _jstd: np.ndarray = None
    _n_iv: int = 0                          # total interventions, exact even after the log is trimmed
    _reasons: dict = field(default_factory=dict)
    _max_drift: float = 0.0
    log: list = field(default_factory=list)

    def calibrate(self, actions):
        """Fit the in-distribution stats from a calibration set. actions: [N>=8, A]."""
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] < 8:
            raise ValueError("calibrate needs a [N, action_dim] array with N >= 8 real samples")
        rng = np.asarray(self.cfg.action_high, float) - np.asarray(self.cfg.action_low, float)
        floor = np.maximum(self.cfg.eps, 1e-3 * np.abs(rng))   # a near-constant dim must not become hypersensitive
        self._mean = a.mean(0)
        self._std = np.maximum(a.std(0), floor)
        d = np.diff(a, axis=0)
        self._jmean = d.mean(0)
        self._jstd = np.maximum(d.std(0), floor)
        self._last_safe = np.clip(a[-1], self.cfg.action_low, self.cfg.action_high)
        return self

    def _pooled_z(self, x, mean, std):
        return float(np.sqrt(np.mean(((x - mean) / std) ** 2)))

    def _safe_out(self):
        if self._last_safe is not None:
            return np.clip(self._last_safe, self.cfg.action_low, self.cfg.action_high)
        return np.zeros(np.asarray(self.cfg.action_low, float).size)

    def _record(self, reasons, drift, jerk, a_in, a_out):
        self._n_iv += 1
        self._max_drift = max(self._max_drift, drift)
        for r in reasons:
            self._reasons[r] = self._reasons.get(r, 0) + 1
        rec = Intervention(self._t, reasons, round(drift, 3), round(jerk, 3),
                           np.asarray(a_in, float).tolist(), np.asarray(a_out, float).tolist())
        self.log.append(rec)
        if len(self.log) > self.cfg.max_log:
            self.log = self.log[-self.cfg.max_log:]
        return rec

    def step(self, action):
        """Vet one proposed action. Returns (safe_action, intervention_or_None)."""
        self._t += 1
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = np.asarray(self.cfg.action_low, float).size

        if a.size != expected:                     # malformed: never crash, treat as unsafe
            out = self._safe_out()
            return out, self._record(["bad_shape"], 0.0, 0.0, a, out)

        reasons = []
        if not np.all(np.isfinite(a)):
            reasons.append("nonfinite")
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        clipped = np.clip(a, self.cfg.action_low, self.cfg.action_high)
        if not np.allclose(clipped, a, atol=1e-9):
            reasons.append("out_of_bounds")

        drift = self._pooled_z(clipped, self._mean, self._std) if self._mean is not None else 0.0
        if drift > self.cfg.drift_thresh:
            reasons.append("drift")

        jerk = 0.0
        if self._last_safe is not None and self._jstd is not None:
            jerk = self._pooled_z(clipped - self._last_safe, self._jmean, self._jstd)
            if jerk > self.cfg.jerk_thresh:
                reasons.append("jerk")

        if reasons:
            out = self._safe_out()
            return out, self._record(reasons, drift, jerk, clipped, out)

        self._last_safe = clipped
        return clipped, None

    def report(self):
        return {"steps": self._t, "interventions": self._n_iv,
                "intervention_rate": round(self._n_iv / max(1, self._t), 4),
                "by_reason": dict(self._reasons), "max_drift": round(self._max_drift, 3),
                "logged": len(self.log)}

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump({"report": self.report(),
                       "interventions": [r.__dict__ for r in self.log]}, f, indent=2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = 7
    cfg = SupervisorConfig(action_low=np.full(A, -1.0), action_high=np.full(A, 1.0))
    sup = Supervisor(cfg).calibrate(rng.normal(0, 0.25, size=(2000, A)).clip(-1, 1))

    for _ in range(200):
        sup.step(rng.normal(0, 0.25, size=A).clip(-1, 1))
    for bad in [np.full(A, np.nan), np.full(A, 5.0), np.zeros(3)]:   # NaN, out-of-bounds, malformed
        _, iv = sup.step(bad)
        print("intervention:", iv.reasons if iv else None)
    print("report:", sup.report())
