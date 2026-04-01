"""
Energy Grid Balancer — FastAPI Server
Uses openenv.core.env_server.create_app for WebSocket-based concurrent sessions.
Falls back to a standalone FastAPI server if openenv-core is not installed.
"""
import json
import os
import sys
import uuid
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GridAction, GridObservation, GridState, GridReward
from server.energy_grid_environment import EnergyGridEnvironment, TASKS, grade_episode

# ── Try to use openenv create_app (preferred) ─────────────────────────────────
try:
    from openenv.core.env_server import create_app

    def _make_env_factory(task_id: str = "easy"):
        def factory():
            return EnergyGridEnvironment(task_id=task_id)
        return factory

    # Default app: easy task (validators will use /reset with task_id override)
    app = create_app(
        lambda: EnergyGridEnvironment(task_id="easy"),   # class passed as factory
        GridAction,
        GridObservation,
        env_name="energy-grid-balancer",
    )
    _USING_OPENENV_CORE = True

except ImportError:
    _USING_OPENENV_CORE = False

# ── Standalone FastAPI fallback (used when openenv-core not installed) ─────────
if not _USING_OPENENV_CORE:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
    import yaml

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)

    app = FastAPI(
        title="Energy Grid Balancer",
        description="OpenEnv-compatible real-world energy grid RL environment.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── HTTP session store ─────────────────────────────────────────────────────
    _sessions: Dict[str, EnergyGridEnvironment] = {}
    MAX_SESSIONS = 50

    def _get_env(sid: str) -> EnergyGridEnvironment:
        env = _sessions.get(sid)
        if env is None:
            raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
        return env

    def _evict():
        if len(_sessions) >= MAX_SESSIONS:
            del _sessions[next(iter(_sessions))]

    # ── Endpoints ──────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_path = os.path.join(_ROOT, "static", "index.html")
        if os.path.exists(html_path):
            return open(html_path).read()
        return HTMLResponse("<h1>⚡ Energy Grid Balancer — OpenEnv</h1><p><a href='/docs'>API Docs</a></p>")

    @app.get("/health")
    async def health():
        return {"status": "ok", "environment": "energy-grid-balancer", "version": "1.0.0",
                "tasks": list(TASKS.keys()), "openenv_core": False}

    @app.post("/reset")
    async def reset(body: dict = None):
        body = body or {}
        task_id = body.get("task_id", "easy")
        seed = body.get("seed", None)
        if task_id not in TASKS:
            raise HTTPException(400, f"Unknown task_id. Choose: {list(TASKS.keys())}")
        sid = body.get("session_id") or str(uuid.uuid4())
        _evict()
        env = EnergyGridEnvironment(task_id=task_id)
        _sessions[sid] = env
        obs = env.reset(seed=seed)
        return {
            "session_id": sid,
            "observation": obs.model_dump(),
            "done": False,
            "reward":0.0,
            "info": {
                "task_id": task_id,
                **{
                    "task_name": TASKS[task_id]["name"],
                    "task_description": TASKS[task_id]["description"],
                }
            }
        }

    @app.post("/step")
    async def step(body: dict):
        sid = body.get("session_id", "")
        env = _get_env(sid)
        if env._done:
            raise HTTPException(400, "Episode done. Call /reset.")
        try:
            action = GridAction(**body.get("action", {}))
        except Exception as e:
            raise HTTPException(400, f"Invalid action: {e}")
        obs = env.step(action)
        reward_obj = GridReward(
            score=float(obs.reward),
            penalty=0.0,
            total=float(obs.reward)
        )
        return {
            "observation": obs.model_dump(),
            "reward": reward_obj.model_dump(),
            "done": bool(obs.done),
            "info": {
                "step": obs.step,
                "battery_soc": obs.battery_soc_pct,
                "grid_frequency_hz": obs.grid_frequency_hz,
                "net_power_kw": obs.net_power_kw,
                "cumulative_cost": obs.cumulative_cost,
            },
        }

    @app.get("/state")
    async def get_state(session_id: str = ""):
        env = _get_env(session_id)
        return {
            "session_id": session_id,
            "observation": env.state.model_dump()
        }

    @app.post("/grade")
    async def grade(body: dict):
        sid = body.get("session_id", "")
        env = _get_env(sid)
        result = env.grade()
        return {"session_id": sid, "task_id": env._task_id, **result}

    @app.get("/tasks")
    async def list_tasks():
        return {"tasks": [
            {k: v for k, v in t.items()
             if k not in ("grader_weights", "seed")}
            for t in TASKS.values()
        ]}

    @app.get("/action_space")
    async def action_space():
        return {
            "type": "Discrete+Continuous",
            "actions": [
                {"action_type": "charge_battery", "description": "Store surplus renewable energy in battery"},
                {"action_type": "sell_to_grid", "description": "Export surplus power to utility grid"},
                {"action_type": "curtail_power", "description": "Reduce generation to prevent overload"},
                {"action_type": "hold", "description": "No active dispatch, default balancing"},
            ],
            "magnitude": {"type": "float", "range": [0.0, 1.0]},
        }

    @app.get("/observation_space")
    async def observation_space():
        return {
            "dimensions": 24,
            "fields": {
                "time": ["hour_of_day", "day_of_week", "season"],
                "generation": ["solar_output_kw", "wind_output_kw", "total_generation_kw"],
                "demand": ["building_demand_kw", "demand_forecast_1h_kw", "demand_forecast_3h_kw"],
                "battery": ["battery_soc_pct", "battery_capacity_kw", "battery_energy_kwh", "battery_max_kwh"],
                "grid": ["grid_frequency_hz", "grid_voltage_pu", "grid_stability_score"],
                "economics": ["grid_buy_price", "grid_sell_price", "carbon_intensity"],
                "balance": ["net_power_kw"],
                "episode": ["step", "max_steps", "cumulative_cost",
                            "cumulative_curtailed_kwh", "cumulative_sold_kwh", "last_action"],
            },
        }

    @app.get("/openenv.yaml", response_class=PlainTextResponse)
    async def openenv_yaml():
        yaml_path = os.path.join(_ROOT, "openenv.yaml")
        if os.path.exists(yaml_path):
            return open(yaml_path).read()
        raise HTTPException(404, "openenv.yaml not found")

    @app.get("/sessions")
    async def sessions_info():
        return {"active_sessions": len(_sessions), "max_sessions": MAX_SESSIONS}

    # ── WebSocket endpoint (OpenEnv spec) ──────────────────────────────────────
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket session endpoint following the OpenEnv protocol.
        Each connection gets its own environment instance.
        """
        await websocket.accept()
        env: Dict[str, EnergyGridEnvironment] = {}

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                method = msg.get("method", "")
                params = msg.get("params", {})

                if method == "reset":
                    task_id = params.get("task_id", "easy")
                    seed = params.get("seed", None)
                    if task_id not in TASKS:
                        await websocket.send_text(json.dumps(
                            {"error": f"Unknown task_id. Choose: {list(TASKS.keys())}"}))
                        continue
                    grid_env = EnergyGridEnvironment(task_id=task_id)
                    env["current"] = grid_env
                    obs = grid_env.reset(seed=seed)
                    await websocket.send_text(json.dumps({
                        "observation": obs.model_dump(),
                        "done": False,
                        "reward": 0.0,
                        "info": {"task_id": task_id},
                    }))

                elif method == "step":
                    if "current" not in env:
                        await websocket.send_text(json.dumps({"error": "Call reset first."}))
                        continue
                    grid_env = env["current"]
                    if grid_env._done:
                        await websocket.send_text(json.dumps({"error": "Episode done. Call reset."}))
                        continue
                    try:
                        action = GridAction(**params.get("action", {}))
                    except Exception as e:
                        await websocket.send_text(json.dumps({"error": f"Invalid action: {e}"}))
                        continue
                    obs = grid_env.step(action)
                    await websocket.send_text(json.dumps({
                        "observation": obs.model_dump(),
                        "done": obs.done,
                        "reward": float(obs.reward),
                        "info": {"step": obs.step, "battery_soc": obs.battery_soc_pct},
                    }))

                elif method == "state":
                    if "current" not in env:
                        await websocket.send_text(json.dumps({"error": "Call reset first."}))
                        continue
                    st = env["current"].state()
                    await websocket.send_text(json.dumps(st.model_dump()))

                elif method == "grade":
                    if "current" not in env:
                        await websocket.send_text(json.dumps({"error": "Call reset first."}))
                        continue
                    result = env["current"].grade()
                    await websocket.send_text(json.dumps(result))

                else:
                    await websocket.send_text(json.dumps(
                        {"error": f"Unknown method '{method}'. Use: reset, step, state, grade"}))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
        )

if __name__ == "__main__":
    main()
