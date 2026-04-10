"""
Energy Grid Balancer — OpenEnv Client
Extends EnvClient for typed WebSocket interaction or provides an HTTP fallback.
Ensures return types are consistent across both modes.
"""
import json
import os
import sys
import requests
from typing import Optional, Dict, Any, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GridAction, GridObservation, GridState

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

# ── Result Wrapper for Fallback Mode ──────────────────────────────────────────

class FallbackStepResult:
    """Mimics openenv.core.client_types.StepResult for HTTP fallback mode."""
    def __init__(self, data: Dict[str, Any]):
        self.observation = GridObservation(**data["observation"])
        self.done = data.get("done", False)
        reward_raw = data.get("reward", 0.0)
        if isinstance(reward_raw, dict):
            self.reward = float(reward_raw.get("total", 0.0))
        else:
            self.reward = float(reward_raw)

# ── Client Implementation ─────────────────────────────────────────────────────

if _HAS_CORE:
    class EnergyGridEnv(EnvClient[GridAction, GridObservation, GridState]):
        """Typed WebSocket client using openenv-core."""
        def _step_payload(self, action: GridAction) -> dict:
            return {
                "action_type": action.action_type,
                "magnitude": action.magnitude,
            }

        def _parse_result(self, payload: dict) -> "StepResult[GridObservation]":
            obs_data = payload.get("observation", {})
            return StepResult(
                observation=GridObservation(**obs_data),
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict) -> GridState:
            return GridState(**payload)

else:
    class EnergyGridEnv:
        """HTTP fallback client when openenv-core is missing."""
        def __init__(self, base_url: str = "http://localhost:7860"):
            self._base = base_url.rstrip("/")
            self._session_id: Optional[str] = None

        @classmethod
        def from_hub(cls, repo_id: str) -> "EnergyGridEnv":
            slug = repo_id.replace("/", "-").lower()
            return cls(base_url=f"https://{slug}.hf.space")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sync(self):
            return self

        def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> FallbackStepResult:
            body = {"task_id": task_id}
            if seed is not None:
                body["seed"] = seed
            
            r = requests.post(f"{self._base}/reset", json=body, timeout=30)
            r.raise_for_status()
            data = r.json()
            self._session_id = data.get("session_id")
            return FallbackStepResult(data)

        def step(self, action: GridAction) -> FallbackStepResult:
            if not self._session_id:
                raise RuntimeError("No active session. Call reset() first.")    
            payload = {
                "session_id": self._session_id,
                "action": action.model_dump() if hasattr(action, "model_dump") else action.__dict__
            }
            r = requests.post(f"{self._base}/step", json=payload, timeout=15)
            r.raise_for_status()
            return FallbackStepResult(r.json())

        def state(self) -> GridState:
            r = requests.get(f"{self._base}/state", params={"session_id": self._session_id}, timeout=10)
            r.raise_for_status()
            return GridState(**r.json())

        def grade(self) -> dict:
            r = requests.post(f"{self._base}/grade", json={"session_id": self._session_id}, timeout=15)
            r.raise_for_status()
            return r.json()