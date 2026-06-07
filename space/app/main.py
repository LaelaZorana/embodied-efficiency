"""embodied-efficiency deploy console: FastAPI + Jinja2 + htmx.

One small server, two pillars on one page:

  - Deploy-compiler: the real-L4 Pareto frontier is serialised to the client, so
    moving a budget slider re-picks the best config and redraws the frontier
    instantly, with no round-trip.
  - Safety supervisor: each "vet" posts to /vet, which runs the *actual*
    governance code from app/supervisor.py against a chosen action and swaps the
    rendered verdict back in. The running intervention log is real state that
    accumulates across clicks, that's the governance trail.

No API keys, no GPU. Footprint, fidelity, and staleness are exact everywhere;
the latencies were measured on an L4 and are reported as fixed data.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.pareto import CONFIGS
from app.supervisor import Supervisor, SupervisorConfig

BASE_DIR = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/LaelaZorana/embodied-efficiency"
HF_THESIS_URL = "https://github.com/LaelaZorana/embodied-efficiency/blob/main/THESIS.md"

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.globals["github_url"] = GITHUB_URL
templates.env.globals["thesis_url"] = HF_THESIS_URL

# ---- the live safety supervisor -------------------------------------------
# One calibrated supervisor for the whole process, so the intervention log is a
# real running governance trail across visitors. A single uvicorn worker serves
# sync routes from a threadpool, so guard the shared state with a lock.
A = 7
_rng = np.random.default_rng(0)
_sup_cfg = SupervisorConfig(action_low=np.full(A, -1.0), action_high=np.full(A, 1.0))
SUP = Supervisor(_sup_cfg).calibrate(_rng.normal(0, 0.15, (1500, A)).clip(-1, 1))
for _ in range(50):                       # warm up so last_safe and the baseline are set
    SUP.step(_rng.normal(0, 0.15, A).clip(-1, 1))
_SUP_LOCK = threading.Lock()

SCENARIOS = {
    "clean": ("Clean action (in distribution)",
              lambda: _rng.normal(0, 0.15, A).clip(-1, 1)),
    "nonfinite": ("NaN in the action (sensor glitch)",
                  lambda: np.full(A, np.nan)),
    "out_of_bounds": ("Out of joint limits (all 5.0)",
                      lambda: np.full(A, 5.0)),
    "drift": ("Drift, far from anything seen (all 0.8)",
              lambda: np.full(A, 0.8)),
}


def _render(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, ctx)


app = FastAPI(title="embodied-efficiency", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return _render(
        request,
        "index.html",
        configs_json=json.dumps(CONFIGS),
        scenarios=[(k, label) for k, (label, _) in SCENARIOS.items()],
    )


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "supervisor_steps": SUP.report()["steps"]})


@app.post("/vet", response_class=HTMLResponse)
def vet(request: Request, scenario: str = Form("clean")):
    label, make = SCENARIOS.get(scenario, SCENARIOS["clean"])
    action = make()
    with _SUP_LOCK:
        out, iv = SUP.step(action)
        report = SUP.report()
    passed = iv is None
    return _render(
        request,
        "partials/_verdict.html",
        label=label,
        passed=passed,
        proposed=np.round(np.asarray(action, float), 2).tolist(),
        sent=np.round(np.asarray(out, float), 2).tolist(),
        reasons=[] if passed else list(iv.reasons),
        drift=0.0 if passed else iv.drift,
        jerk=0.0 if passed else iv.jerk,
        report=report,
    )
