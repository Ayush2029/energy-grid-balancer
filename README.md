---
title: Energy Grid Balancer OpenEnv
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - energy
  - renewable
  - reinforcement-learning
  - agent
  - real-world
license: mit
---

# ⚡ Energy Grid Balancer — OpenEnv

> A **real-world** renewable energy grid management environment for AI agent training and evaluation.
> Built for the [OpenEnv Hackathon](http://meta-pytorch.org/OpenEnv) by Meta & Hugging Face.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-00d4ff?style=flat-square)](http://meta-pytorch.org/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🌍 Problem Statement

Modern power grids are increasingly dependent on solar and wind — both inherently intermittent.
Unlike traditional power plants, these sources cannot be switched on demand.
Every 10 minutes an AI agent must decide:

- **Charge battery** — store surplus renewable energy for later
- **Sell to grid** — export surplus at market price for revenue
- **Curtail power** — safely dump excess to protect equipment
- **Hold** — let battery auto-discharge to cover demand shortfall

The goal: minimise cost and wasted energy while keeping grid frequency stable.

---

## 🔌 Observation Space (24 dimensions)

| Category | Fields |
|----------|--------|
| **Time** | `hour_of_day`, `day_of_week`, `season` |
| **Generation** | `solar_output_kw`, `wind_output_kw`, `total_generation_kw` |
| **Demand** | `building_demand_kw`, `demand_forecast_1h_kw`, `demand_forecast_3h_kw` |
| **Battery** | `battery_soc_pct`, `battery_energy_kwh`, `battery_max_kwh`, `battery_capacity_kw` |
| **Grid** | `grid_frequency_hz`, `grid_voltage_pu`, `grid_stability_score` |
| **Economics** | `grid_buy_price`, `grid_sell_price`, `carbon_intensity` |
| **Balance** | `net_power_kw` |
| **Episode** | `step`, `max_steps`, `cumulative_cost`, `cumulative_curtailed_kwh`, `cumulative_sold_kwh`, `last_action` |

Grid frequency is modelled via a swing-equation approximation — frequency deviates
from 50 Hz based on real-time power imbalance, giving the agent a physical stability signal.

---

## 🎮 Action Space

```json
{
  "action_type": "charge_battery | sell_to_grid | curtail_power | hold",
  "magnitude": 0.0
}
```

| Action | Effect |
|--------|--------|
| `charge_battery` | Store surplus in battery (94 % charge efficiency) |
| `sell_to_grid` | Export surplus at market price (restricted in storms) |
| `curtail_power` | Safely dump surplus; penalised by task multiplier |
| `hold` | No-op; battery auto-discharges to meet demand |

---

## 📊 Reward Function (shaped, −2 to +2 per step)

```
reward = stability_reward         # 0 to +0.4  — frequency within ±0.2 Hz
       + economic_reward          # proportional to avoided import cost
       + curtailment_penalty      # −kWh wasted × task multiplier × 0.05
       + frequency_penalty        # cascade penalty beyond ±0.2 Hz
       + battery_efficiency       # +0.1 (SoC 30–80 %), −0.15 (outside)
       + action_reward            # context-specific bonus/penalty
```

Partial-progress signals appear every step so the agent can learn from the
stability axis before it has mastered cost minimisation.

---

## 🎯 Tasks

### `easy` — Sunny Day Balancing
| Property | Value |
|----------|-------|
| Scenario | 80 kW solar, clear summer day, small commercial building |
| Episode | 48 steps × 10 min = **8 hours** |
| Battery | 100 kWh / 50 kW rate |
| Volatility | 0.1 (low) |
| Grader weights | Cost 35 % · Curtailment 30 % · Stability 15 % · Battery 10 % · Completion 10 % |

### `medium` — Mixed Renewables District
| Property | Value |
|----------|-------|
| Scenario | 120 kW solar + 60 kW wind, full 24-hour cycle |
| Episode | 144 steps × 10 min = **24 hours** |
| Battery | 150 kWh / 75 kW rate |
| Volatility | 0.35 (medium — cloud events, wind gusts, evening demand peaks) |
| Grader weights | Cost 25 % · Curtailment 25 % · Stability 25 % · Battery 15 % · Completion 10 % |

### `hard` — Storm Resilience Challenge
| Property | Value |
|----------|-------|
| Scenario | 150 kW solar + 100 kW wind, 3-day storm window, critical infrastructure |
| Episode | 432 steps × 10 min = **72 hours** |
| Battery | 120 kWh / 60 kW rate (undersized) |
| Volatility | 0.75 (high — intermittent generation, export restrictions, frequency cascades) |
| Grader weights | Stability 35 % · Cost 20 % · Curtailment 20 % · Battery 15 % · Completion 10 % |

---

## 🏆 Grading (deterministic, 0.0–1.0)

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| Curtailment score | 30 % | 25 % | 20 % |
| Cost efficiency | 35 % | 25 % | 20 % |
| Grid stability | 15 % | 25 % | **35 %** |
| Battery health | 10 % | 15 % | 15 % |
| Completion | 10 % | 10 % | 10 % |

---

## 🚀 Hackathon Workflow (exact commands)

### Step 1 — Install the CLI

```bash
pip install "openenv-core>=0.2.1"
# or from source:
pip install "git+https://github.com/meta-pytorch/OpenEnv.git"
```

### Step 2 — Scaffold (already done — this repo IS the scaffold)

```bash
# If starting fresh:
openenv init energy_grid_balancer
```

---

### Step 3 — Build (local test)

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/energy-grid-balancer
cd energy-grid-balancer
````

### ⚠️ Python Setup (Mac/Linux users)

If `python` points to Python 2 on your system, use:

```bash
alias python=python3
alias pip=pip3
```

OR (recommended — isolated environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 🔧 Setup environment

```bash
cp .env.example .env   # fill in your API key
```

### ▶️ Run locally

```bash
# Run with OpenEnv
openenv serve   # starts FastAPI on :8000

# OR run with Docker
docker build -t energy-grid-balancer .
docker run -p 7860:7860 --env-file .env energy-grid-balancer
```

### Step 4 — Test locally

```bash
# Validate the environment structure
openenv validate --verbose

# Run the baseline inference script
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

Expected output:

```
  ⚡  ENERGY GRID BALANCER — BASELINE INFERENCE
  easy      [████████████████████░░░░]  0.9700
  medium    [████████████████░░░░░░░░]  0.8329
  hard      [██████████████░░░░░░░░░░]  0.7474
  AVERAGE   : 0.8501
```

### Step 5 — Deploy to HuggingFace Spaces

```bash
openenv push --repo-id YOUR_USERNAME/energy-grid-balancer
# or manually:
git push  # HF Space auto-builds from Dockerfile
```

### Step 6 — Submit

Paste your Space URL: `https://YOUR_USERNAME-energy-grid-balancer.hf.space`

---

## 🤖 Using the Client

```python
# HTTP client (no openenv-core needed)
import requests

BASE = "https://YOUR_USERNAME-energy-grid-balancer.hf.space"

r = requests.post(f"{BASE}/reset", json={"task_id": "medium"})
sid = r.json()["session_id"]
obs = r.json()["observation"]

done = False
while not done:
    action = {"action_type": "charge_battery", "magnitude": 0.7}
    r = requests.post(f"{BASE}/step", json={"session_id": sid, "action": action})
    obs  = r.json()["observation"]
    done = r.json()["done"]

score = requests.post(f"{BASE}/grade", json={"session_id": sid}).json()["score"]
print(f"Score: {score:.3f}")
```

```python
# Typed WebSocket client (openenv-core installed)
from client import EnergyGridEnv
from models import GridAction

with EnergyGridEnv.from_hub("YOUR_USERNAME/energy-grid-balancer").sync() as env:
    obs = env.reset(task_id="medium")
    while not obs.done:
        action = GridAction(action_type="charge_battery", magnitude=0.8)
        obs = env.step(action)
    result = env.grade()
    print(f"Score: {result['score']:.3f}")
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check (must return 200) |
| `POST` | `/reset` | Start new episode; body: `{task_id, session_id?, seed?}` |
| `POST` | `/step` | Execute action; body: `{session_id, action}` |
| `GET`  | `/state?session_id=...` | Full state dict |
| `POST` | `/grade` | Grade episode (0.0–1.0); body: `{session_id}` |
| `GET`  | `/tasks` | List tasks with metadata |
| `GET`  | `/action_space` | Action space definition |
| `GET`  | `/observation_space` | Observation space definition |
| `GET`  | `/openenv.yaml` | OpenEnv YAML metadata |
| `WS`   | `/ws` | WebSocket: `{method, params}` protocol |
| `GET`  | `/docs` | FastAPI Swagger UI |

---

## 🔬 Physical Models

| Model | Implementation |
|-------|---------------|
| Solar | Bell-curve clearsky irradiance (peak noon) + stochastic cloud events |
| Wind | Weibull wind speed + 3-zone Betz power curve (cut-in 3 m/s, rated 12, cut-out 25) |
| Demand | Commercial building daily profile, weekday/weekend, stochastic noise |
| Battery | 94 % charge/discharge efficiency, SoC bounds 10–95 % |
| Frequency | Swing-equation approximation: imbalance → Hz deviation → stability score |
| Pricing | Time-of-use: peak $0.25 buy / $0.18 sell; off-peak $0.12 / $0.08 |

---

## 📁 Project Structure

```
energy-grid-balancer/          ← openenv init output
├── __init__.py
├── models.py                  ← GridAction, GridObservation, GridState (Pydantic)
├── client.py                  ← EnergyGridEnv typed WebSocket client
├── inference.py               ← Baseline LLM agent (OpenAI client)
├── openenv.yaml               ← OpenEnv metadata
├── pyproject.toml             ← uv-compatible dependency manifest
├── Dockerfile                 ← multi-stage, openenv-base compatible
├── .env.example               ← credentials template
├── .gitignore
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py                 ← FastAPI + WebSocket server (create_app)
│   ├── energy_grid_environment.py  ← core simulation (extends Environment)
│   └── requirements.txt
└── static/
    └── index.html             ← interactive dashboard
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes (inference) | LLM API endpoint |
| `MODEL_NAME` | Yes (inference) | Model identifier |
| `HF_TOKEN` | Yes (inference) | OpenAI / HuggingFace API key |
| `ENV_BASE_URL` | No | Server URL (default: `http://localhost:7860`) |
| `PORT` | No | Server port (default: `7860`) |

---

## 🧪 Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

---

## 📜 Baseline Scores

Achieved with `gpt-4o-mini` via OpenAI API:

| Task | Score | Agent |
|------|-------|-------|
| easy | 0.986 | LLM (rule-assisted) |
| medium | 0.869 | LLM |
| hard | 0.749 | LLM |
| **Average** | **0.868** | — |

---

*Built for the OpenEnv Challenge — Real-world AI agent environment for renewable energy grid management*
