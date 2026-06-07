"""
Runtime safety supervisor for an embodied or agentic policy.

A policy that's fast isn't the same as a policy you can leave running on its own.
This sits between the policy and the actuator, watches every action it proposes,
and when an action drifts somewhere the policy was never calibrated for, it swaps
in a safe fallback and writes down why. That log is the governance trail: an
on-call engineer or a regulator can read exactly when the policy got overridden
and what tripped it.

What it checks, on every action:
  - finite: no NaN or Inf ever reaches a motor.
  - in-bounds: every dimension stays inside the action limits.
  - drift (OOD): how far this action sits from the calibration set, as a per-dim
    z-score pooled into one distance. A high distance means the policy is acting
    outside the slice it was actually checked on.
  - jerk: how big the jump is from the last action, against the calibration jerk,
    because a smooth policy that suddenly lurches is a red flag even when the
    action itself looks in-distribution.

When a check trips, the supervisor returns a safe action (hold the last accepted
action, clipped to limits) and appends an intervention record. Nothing here needs
a GPU. It's the trust layer that rides on top of the efficient policy, and it's
the second half of the thesis: efficiency gets the model onto the robot, this is
what lets it stay there.
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


@dataclass
class Intervention:
    step: int
    reasons: list                   # which checks tripped, e.g. ["drift", "jerk"]
    drift: float
    jerk: float
    action_in: list                 # what the policy proposed (sanitized to finite)
    action_out: list                # what was actually sent


@dataclass
class Supervisor:
    cfg: SupervisorConfig
    _t: int = 0
    _last_safe: np.ndarray = None
    _mean: np.ndarray = None
    _std: np.ndarray = None
    _jmean: np.ndarray = None
    _jstd: np.ndarray = None
    log: list = field(default_factory=list)

    def calibrate(self, actions):
        """Fit the in-distribution stats from a calibration set. actions: [N, A]."""
        a = np.asarray(actions, dtype=np.float64)
        self._mean, self._std = a.mean(0), a.std(0) + self.cfg.eps
        d = np.diff(a, axis=0)
        self._jmean, self._jstd = d.mean(0), d.std(0) + self.cfg.eps
        self._last_safe = np.clip(a[-1], self.cfg.action_low, self.cfg.action_high)
        return self

    def _pooled_z(self, x, mean, std):
        return float(np.sqrt(np.mean(((x - mean) / std) ** 2)))

    def step(self, action):
        """Vet one proposed action. Returns (safe_action, intervention_or_None)."""
        self._t += 1
        a = np.asarray(action, dtype=np.float64)
        reasons = []

        finite = np.all(np.isfinite(a))
        if not finite:
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
            # don't trust a flagged action; hold the last accepted one (still clipped)
            out = np.clip(self._last_safe, self.cfg.action_low, self.cfg.action_high) \
                if self._last_safe is not None else clipped
            rec = Intervention(self._t, reasons, round(drift, 3), round(jerk, 3),
                               clipped.tolist(), out.tolist())
            self.log.append(rec)
            return out, rec

        self._last_safe = clipped
        return clipped, None

    def report(self):
        reasons = {}
        for r in self.log:
            for name in r.reasons:
                reasons[name] = reasons.get(name, 0) + 1
        return {"steps": self._t, "interventions": len(self.log),
                "intervention_rate": round(len(self.log) / max(1, self._t), 4),
                "by_reason": reasons,
                "max_drift": round(max((r.drift for r in self.log), default=0.0), 3)}

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump({"report": self.report(),
                       "interventions": [r.__dict__ for r in self.log]}, f, indent=2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = 7
    cfg = SupervisorConfig(action_low=np.full(A, -1.0), action_high=np.full(A, 1.0))
    sup = Supervisor(cfg).calibrate(rng.normal(0, 0.25, size=(2000, A)).clip(-1, 1))

    # a normal rollout, then three bad actions: NaN, far out-of-distribution, a hard lurch
    sent = 0
    for _ in range(200):
        a, iv = sup.step(rng.normal(0, 0.25, size=A).clip(-1, 1))
        sent += iv is None
    for bad in [np.full(A, np.nan), np.full(A, 5.0), np.array([0.9] + [-0.9] * (A - 1))]:
        _, iv = sup.step(bad)
        print("intervention:", iv.reasons if iv else None)
    print("report:", sup.report())
