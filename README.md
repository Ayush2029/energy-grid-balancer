# Energy Grid Balancer — OpenEnv RL Environment

> A real-world inspired renewable energy grid simulation environment for Reinforcement Learning agents and LLM-based decision systems.

---

## Overview

**Energy Grid Balancer** is an OpenEnv-compatible simulation environment designed to model a **renewable energy microgrid**. It enables agents (rule-based, RL, or LLM-powered) to make intelligent decisions for balancing:

* Energy generation (solar + wind)
* Demand consumption
* Battery storage
* Grid interaction (buy/sell/curtail)

---

## Problem Statement

Modern renewable grids face challenges like:

* Intermittent generation (solar/wind variability)
* Demand fluctuations
* Battery constraints
* Grid instability (frequency deviations)

This project simulates these real-world constraints and challenges an agent to:

> **Maximize efficiency, minimize cost, and maintain grid stability.**

---

## 🧠 Key Features

### 1. Custom RL Environment

* Fully configurable **OpenEnv-compatible environment**
* Supports:

  * `reset()`, `step()`, `state()`, `grade()`
* Designed for:

  * RL agents
  * LLM agents
  * Hybrid strategies

---

### 2. Multi-Difficulty Tasks

Three real-world inspired scenarios:

| Task   | Description                       | Duration |
| ------ | --------------------------------- | -------- |
| EASY   | Sunny day, solar only             | 8 hours  |
| MEDIUM | Solar + wind + demand variation   | 24 hours |
| HARD   | Storm conditions, high volatility | 72 hours |

---

### 3. Hybrid AI Agent (LLM + Rules)

* Combines:

  * Rule-based heuristics
  * LLM decision making 
* Optimizes:

  * Cost
  * Curtailment
  * Stability
  * Battery health

---

### 4. FastAPI Server (OpenEnv Compatible)

* REST + WebSocket support
* Session-based environment handling
* Scalable concurrent simulations

Endpoints:

* `/reset`
* `/step`
* `/state`
* `/grade`
* `/tasks`


---

### 5. Interactive Web UI

A modern dashboard to visualize:

* Real-time power flow
* Battery state
* Grid frequency & stability
* Cost & rewards
* Performance metrics

---

### 6. Typed Models (Robust API)

* Strongly typed:

  * `GridAction`
  * `GridObservation`
  * `GridState`

* Validation via Pydantic (with fallback)


---

### 7. Client SDK (WebSocket + HTTP)

* Works with:

  * OpenEnv core (WebSocket)
  * HTTP fallback mode

---

## System Architecture

```
                ┌────────────────────┐
                │   Web UI (HTML)    │
                └────────┬───────────┘
                         │ REST API
                ┌────────▼───────────┐
                │   FastAPI Server   │
                └────────┬───────────┘
                         │
                ┌────────▼───────────┐
                │ EnergyGridEnv      │
                │ (Simulation Core)  │
                └────────┬───────────┘
                         │
        ┌────────────────┴──────────────┐
        │                               │
┌───────▼────────┐             ┌────────▼────────┐
│ Rule-based AI  │             │ LLM Agent       │
└────────────────┘             └─────────────────┘
```

---

## Project Structure

```
energy-grid-balancer/
│
├── static/
│   └── index.html              # Interactive UI Dashboard
│
├── server/
│   ├── __init__.py             # Server package initializer
│   ├── app.py                  # FastAPI server entrypoint
│   ├── energy_grid_environment.py  # Core simulation environment
│   └── requirements.txt        # Server-specific dependencies
│
├── __init__.py                 # Package initializer
├── client.py                   # OpenEnv client (WS + HTTP fallback)
├── inference.py                # Hybrid AI agent (LLM + rules)
├── models.py                   # Data models (Action, Observation, State)
├── openenv.yaml                # Benchmark configuration
├── baseline_results.json       # Baseline evaluation results
├── pyproject.toml              # Project metadata & build config
├── uv.lock                     # Dependency lock file
├── Dockerfile                  # Containerization setup
├── README.md                   # Project documentation
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/energy-grid-balancer.git
cd energy-grid-balancer
```

---

## 2. Run Backend using Docker (Recommended)

Build and start the environment server:

```bash
docker build -t energy-env .
docker run -p 7860:7860 energy-env
```

This will:

* Start the **FastAPI server**
* Host the environment at: `http://localhost:7860`

---

## 3. Setup Agent Environment (Local)

Open a **new terminal** and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### Install Required Dependencies

```bash
pip install python-dotenv openai requests
```

---

### Configure Environment Variables

Create a `.env` file in root:

```env
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=your_api_key
ENV_BASE_URL=http://localhost:7860
```

Or export manually (On MAC use export && set for Windows User) :

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_api_key
export ENV_BASE_URL=http://localhost:7860
```

---

## Running the System

### Step 1: Ensure Server is Running (Docker Terminal)

```
http://localhost:7860
```

---

### Step 2: Run AI Agent (Second Terminal)

```bash
python inference.py
```

---

## Workflow Summary

```
Terminal 1 (Docker)
    └── Runs FastAPI + Environment

Terminal 2 (.venv)
    └── Runs LLM Agent (inference.py)
```

---

## Action Space

| Action           | Description       |
| ---------------- | ----------------- |
| `charge_battery` | Store energy      |
| `sell_to_grid`   | Export energy     |
| `curtail_power`  | Reduce generation |
| `hold`           | Do nothing        |

---

## Observation Space

Includes:

* Time: hour, day, season
* Generation: solar, wind
* Demand forecasts
* Battery state
* Grid metrics:

  * Frequency
  * Stability
* Economics:

  * Buy/sell price
  * Carbon intensity

---

## Scoring System

Final score ∈ [0,1], based on:

* Curtailment minimization
* Cost efficiency
* Grid stability
* Battery health
* Task completion


---

## Baseline Performance

```json
Average Score: 0.9459
```

| Task   | Score  |
| ------ | ------ |
| Easy   | 0.9999 |
| Medium | 0.9502 |
| Hard   | 0.8875 |

---

## Contributors

* **Ayush Dharaiya** 
* **Harvy Doshi**
* **Archi PaTEL**

---

## License

MIT License

---
