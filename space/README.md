---
title: embodied-efficiency
emoji: "🤖"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
short_description: Deploy console for VLAs, compiler + safety supervisor
---

# embodied-efficiency, the deploy console

An interactive console for getting a vision-language-action (VLA) model onto the robot and keeping it safe once it's there. Two pillars, one page, no API key and no GPU needed.

- **Deploy-compiler.** Set a latency and footprint budget and it picks the best config off a Pareto frontier measured on a real L4, redrawing the frontier live as you move a slider. Action-chunking, precision, and flow steps are the levers; latency was measured on hardware, footprint and fidelity compute anywhere.
- **Safety supervisor.** Throw an action at the runtime trust layer (a clean one, a NaN, one out of joint limits, one that's drifted far from anything calibrated) and watch it pass the good ones and hold a safe fallback for the rest. This runs the actual `supervisor.py` from the repo, and the intervention log is a real running governance trail.

Built with FastAPI + Jinja2 + htmx and a vendored, offline Tailwind build, so the whole thing is one small container with no build step and no network calls.

Code and the full write-up: [github.com/LaelaZorana/embodied-efficiency](https://github.com/LaelaZorana/embodied-efficiency)
