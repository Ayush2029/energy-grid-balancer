"""
Energy Grid Balancer — OpenEnv Models
Pydantic models for Action, Observation, and State.
Falls back to plain Python classes if pydantic is not installed.
"""
from typing import Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

try:
    from openenv.core.env_server.types import Action as _CoreAction, Observation as _CoreObs, State as _CoreState
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

VALID_ACTIONS = {"charge_battery", "sell_to_grid", "curtail_power", "hold"}

# ── Choose base classes ───────────────────────────────────────────────────────
if _PYDANTIC and _HAS_CORE:
    _BASE_ACTION = _CoreAction
    _BASE_OBS    = _CoreObs
    _BASE_STATE  = _CoreState
elif _PYDANTIC:
    _BASE_ACTION = BaseModel
    _BASE_OBS    = BaseModel
    _BASE_STATE  = BaseModel
else:
    class _StdBase:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    _BASE_ACTION = _StdBase
    _BASE_OBS    = _StdBase
    _BASE_STATE  = _StdBase

# ── GridAction ────────────────────────────────────────────────────────────────

if _PYDANTIC:
    class GridAction(_BASE_ACTION):
        action_type: Literal["charge_battery", "sell_to_grid", "curtail_power", "hold"] = Field(
            default="hold", description="Which grid action to take"
        )
        magnitude:   float = Field(default=0.5, ge=0.0, le=1.0,
            description="Fraction of available capacity to apply (0.0–1.0)")

        @field_validator("action_type")
        @classmethod
        def _check_action(cls, v):
            if v not in VALID_ACTIONS:
                raise ValueError(f"action_type must be one of {VALID_ACTIONS}")
            return v

        class Config:
            extra = "forbid"
else:
    class GridAction(_BASE_ACTION):
        def __init__(self, action_type="hold", magnitude=0.5, **_):
            if action_type not in VALID_ACTIONS:
                raise ValueError(f"action_type must be one of {VALID_ACTIONS}")
            self.action_type = action_type
            self.magnitude = max(0.0, min(1.0, float(magnitude)))
        def model_dump(self):
            return {"action_type": self.action_type, "magnitude": self.magnitude}

# ── GridObservation ───────────────────────────────────────────────────────────

_OBS_DEFAULTS = dict(
    hour_of_day=0.0, day_of_week=0, season="summer",
    solar_output_kw=0.0, wind_output_kw=0.0, total_generation_kw=0.0,
    building_demand_kw=0.0, demand_forecast_1h_kw=0.0, demand_forecast_3h_kw=0.0,
    battery_soc_pct=50.0, battery_capacity_kw=50.0, battery_energy_kwh=50.0, battery_max_kwh=100.0,
    grid_frequency_hz=50.0, grid_voltage_pu=1.0, grid_stability_score=1.0,
    grid_buy_price=0.20, grid_sell_price=0.15, carbon_intensity=250.0,
    net_power_kw=0.0,
    step=0, max_steps=48, cumulative_cost=0.0, cumulative_curtailed_kwh=0.0, cumulative_sold_kwh=0.0,
    last_action=None, done=False, reward=0.0, task_id="easy",
)

if _PYDANTIC:
    class GridObservation(_BASE_OBS):
        hour_of_day:               float = Field(0.0)
        day_of_week:               int   = Field(0)
        season:                    Literal["summer", "winter", "monsoon"] = Field("summer")
        solar_output_kw:           float = Field(0.0)
        wind_output_kw:            float = Field(0.0)
        total_generation_kw:       float = Field(0.0)
        building_demand_kw:        float = Field(0.0)
        demand_forecast_1h_kw:     float = Field(0.0)
        demand_forecast_3h_kw:     float = Field(0.0)
        battery_soc_pct:           float = Field(50.0, ge=0.0, le=100.0)
        battery_capacity_kw:       float = Field(50.0)
        battery_energy_kwh:        float = Field(50.0)
        battery_max_kwh:           float = Field(100.0)
        grid_frequency_hz:         float = Field(50.0, ge=45.0, le=55.0)
        grid_voltage_pu:           float = Field(1.0, ge=0.8, le=1.2)
        grid_stability_score:      float = Field(1.0)
        grid_buy_price:            float = Field(0.20)
        grid_sell_price:           float = Field(0.15)
        carbon_intensity:          float = Field(250.0)
        net_power_kw:              float = Field(0.0)
        step:                      int   = Field(0)
        max_steps:                 int   = Field(48)
        cumulative_cost:           float = Field(0.0)
        cumulative_curtailed_kwh:  float = Field(0.0)
        cumulative_sold_kwh:       float = Field(0.0)
        last_action:               Optional[Literal["charge_battery", "sell_to_grid", "curtail_power", "hold"]] = Field(None)
        done:                      bool  = Field(False)
        reward:                    float = Field(0.0)
        task_id:                   str   = Field("easy")

        class Config:
            extra = "forbid"
else:
    class GridObservation(_BASE_OBS):
        def __init__(self, **kwargs):
            for k, v in _OBS_DEFAULTS.items():
                setattr(self, k, kwargs.get(k, v))
        def model_dump(self):
            return {k: getattr(self, k) for k in _OBS_DEFAULTS}

# ── GridState ─────────────────────────────────────────────────────────────────

if _PYDANTIC:
    class GridState(_BASE_STATE):
        episode_id:               str   = Field("")
        step_count:               int   = Field(0)
        task_id:                  str   = Field("easy")
        episode_done:             bool  = Field(False)
        history_length:           int   = Field(0)
        battery_soc_pct:          float = Field(50.0, ge=0.0, le=100.0)
        cumulative_cost:          float = Field(0.0)
        cumulative_curtailed_kwh: float = Field(0.0)
        cumulative_sold_kwh:      float = Field(0.0)

        class Config:
            extra = "forbid"
else:
    class GridState(_BASE_STATE):
        def __init__(self, episode_id="", step_count=0, task_id="easy",
                     episode_done=False, history_length=0, battery_soc_pct=50.0,
                     cumulative_cost=0.0, cumulative_curtailed_kwh=0.0,
                     cumulative_sold_kwh=0.0, **_):
            self.episode_id               = episode_id
            self.step_count               = step_count
            self.task_id                  = task_id
            self.episode_done             = episode_done
            self.history_length           = history_length
            self.battery_soc_pct          = battery_soc_pct
            self.cumulative_cost          = cumulative_cost
            self.cumulative_curtailed_kwh = cumulative_curtailed_kwh
            self.cumulative_sold_kwh      = cumulative_sold_kwh
        def model_dump(self):
            return self.__dict__

if _PYDANTIC:
    class GridReward(_BASE_STATE): 
        score:   float = Field(0.0)
        penalty: float = Field(0.0)
        total:   float = Field(0.0)

        class Config:
            extra = "forbid"
else:
    class GridReward(_BASE_STATE):
        def __init__(self, score=0.0, penalty=0.0, total=0.0, **_):
            self.score = score
            self.penalty = penalty
            self.total = total
        def model_dump(self):
            return self.__dict__