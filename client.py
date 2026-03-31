"""
Energy Grid Balancer — OpenEnv Client

Extends EnvClient[GridAction, GridObservation, GridState] for typed WebSocket
interaction with the environment server.

Usage:
    from energy_grid_balancer import EnergyGridEnv
    from models import GridAction

    with EnergyGridEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        obs = result.observation
        result = env.step(GridAction(action_type="charge_battery", magnitude=0.8))
        print(result.observation.battery_soc_pct)

    # Or from HF Hub:
    with EnergyGridEnv.from_hub("your-username/energy-grid-balancer").sync() as env:
        ...
"""
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GridAction, GridObservation, GridState

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult

    class EnergyGridEnv(EnvClient[GridAction, GridObservation, GridState]):
        """
        Typed OpenEnv client for the Energy Grid Balancer environment.
        Connects via WebSocket for persistent, low-latency sessions.
        """

        def _step_payload(self, action: GridAction) -> dict:
            """Serialize action for transmission."""
            return {
                "action_type": action.action_type,
                "magnitude": action.magnitude,
            }

        def _parse_result(self, payload: dict) -> "StepResult[GridObservation]":
            """Deserialize server response into typed StepResult."""
            obs_data = payload.get("observation", {})
            obs = GridObservation(**obs_data)
            return StepResult(
                observation=obs,
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict) -> GridState:
            """Deserialize state payload into GridState."""
            return GridState(**payload)

except ImportError:
    # Fallback HTTP client (no openenv-core)
    import requests

    class EnergyGridEnv:
        """
        HTTP fallback client for the Energy Grid Balancer.
        Used when openenv-core is not installed.
        """

        def __init__(self, base_url: str = "http://localhost:8000"):
            self._base = base_url.rstrip("/")
            self._session_id: Optional[str] = None

        @classmethod
        def from_hub(cls, repo_id: str) -> "EnergyGridEnv":
            """Connect to a HuggingFace Spaces deployment."""
            # HF Spaces URL format
            slug = repo_id.replace("/", "-").lower()
            url = f"https://{slug}.hf.space"
            return cls(base_url=url)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sync(self):
            return self

        def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> GridObservation:
            body = {"task_id": task_id}
            if seed is not None:
                body["seed"] = seed
            r = requests.post(f"{self._base}/reset", json=body, timeout=30)
            r.raise_for_status()
            data = r.json()
            self._session_id = data["session_id"]
            return GridObservation(**data["observation"])

        def step(self, action: GridAction) -> GridObservation:
            r = requests.post(f"{self._base}/step", json={
                "session_id": self._session_id,
                "action": {"action_type": action.action_type, "magnitude": action.magnitude},
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            return GridObservation(**data["observation"])

        def state(self) -> GridState:
            r = requests.get(f"{self._base}/state", params={"session_id": self._session_id}, timeout=10)
            r.raise_for_status()
            return GridState(**r.json())

        def grade(self) -> dict:
            r = requests.post(f"{self._base}/grade", json={"session_id": self._session_id}, timeout=15)
            r.raise_for_status()
            return r.json()
