"""
Energy Grid Balancer — Core Environment Implementation
Extends openenv.core.env_server.interfaces.Environment.
"""
import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False
    class State:
        def __init__(self, episode_id="", step_count=0):
            self.episode_id = episode_id
            self.step_count = step_count

    class Environment:
        pass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import GridAction, GridObservation, GridState, VALID_ACTIONS

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "task_id": "easy",
        "name": "Sunny Day Balancing",
        "difficulty": "easy",
        "description": "Small commercial building with 80 kW solar on a clear summer day. 8-hour episode.",
        "episode_length_steps": 48,
        "time_step_minutes": 10,
        "battery_capacity_kwh": 100.0,
        "battery_max_rate_kw": 50.0,
        "solar_capacity_kw": 80.0,
        "wind_capacity_kw": 0.0,
        "building_max_demand_kw": 60.0,
        "grid_sell_enabled": True,
        "curtailment_penalty_multiplier": 1.0,
        "volatility": 0.1,
        "seed": 42,
        "grader_weights": {"curtail": 0.30, "cost": 0.35, "stab": 0.15, "batt": 0.10, "comp": 0.10},
    },
    "medium": {
        "task_id": "medium",
        "name": "Mixed Renewables District",
        "difficulty": "medium",
        "description": "Solar + wind district over a full 24-hour cycle. Cloud events, wind gusts.",
        "episode_length_steps": 144,
        "time_step_minutes": 10,
        "battery_capacity_kwh": 150.0,
        "battery_max_rate_kw": 75.0,
        "solar_capacity_kw": 120.0,
        "wind_capacity_kw": 60.0,
        "building_max_demand_kw": 140.0,
        "grid_sell_enabled": True,
        "curtailment_penalty_multiplier": 1.5,
        "volatility": 0.35,
        "seed": 123,
        "grader_weights": {"curtail": 0.25, "cost": 0.25, "stab": 0.25, "batt": 0.15, "comp": 0.10},
    },
    "hard": {
        "task_id": "hard",
        "name": "Storm Resilience Challenge",
        "difficulty": "hard",
        "description": "Critical microgrid during a 3-day storm. Intermittent generation, export restrictions.",
        "episode_length_steps": 432,
        "time_step_minutes": 10,
        "battery_capacity_kwh": 120.0,
        "battery_max_rate_kw": 60.0,
        "solar_capacity_kw": 150.0,
        "wind_capacity_kw": 100.0,
        "building_max_demand_kw": 200.0,
        "grid_sell_enabled": True,
        "curtailment_penalty_multiplier": 2.0, 
        "volatility": 0.75,
        "seed": 777,
        "grader_weights": {"curtail": 0.20, "cost": 0.20, "stab": 0.35, "batt": 0.15, "comp": 0.10},
    },
}

FREQ_NOM = 50.0
VOLT_NOM = 1.0
CHRG_EFF = 0.98 
DSCH_EFF = 0.98 
SOC_MIN = 20.0  
SOC_MAX = 90.0  

def grade_episode(task_id: str, history: List[Dict]) -> Dict[str, Any]:
    if not history:
        return {"score": 0.0, "breakdown": {}, "feedback": "No steps recorded."}

    cfg = TASKS[task_id]
    n = len(history)
    mx = cfg["episode_length_steps"]
    ts_h = cfg["time_step_minutes"] / 60.0
    w = cfg["grader_weights"]
    cut = sum(h.get("curtailed_kw", 0) * ts_h for h in history)
    gen = sum((h.get("solar_kw", 0) + h.get("wind_kw", 0)) * ts_h for h in history)
    cr = cut / max(gen, 1)
    cs = max(0.0, 1.0 - cr * 2)
    cost = sum(h.get("cost", 0) for h in history)
    bl = n * 0.25 * ts_h * cfg["building_max_demand_kw"] * 0.35 
    es = max(0.0, min(1.0, 1.0 - cost / max(bl, 1)))
    fv = sum(1 for h in history if abs(h.get("freq_hz", 50) - 50) > 0.2)
    ss = max(0.0, 1.0 - fv / max(n, 1))
    soc_list = [h.get("battery_soc", 50) for h in history]
    bs = sum(1 for s in soc_list if 20 <= s <= 90) / max(n, 1)
    cmp = n / mx
    raw_sc = w["curtail"] * cs + w["cost"] * es + w["stab"] * ss + w["batt"] * bs + w["comp"] * cmp
    sc = round(max(0.0001, min(0.9999, raw_sc)), 4)

    gl = "A" if sc >= 0.85 else "B" if sc >= 0.70 else "C" if sc >= 0.55 else "D"
    return {
        "score": sc,
        "breakdown": {
            "curtailment_score": round(cs, 4),
            "cost_efficiency_score": round(es, 4),
            "grid_stability_score": round(ss, 4),
            "battery_health_score": round(bs, 4),
            "completion_score": round(cmp, 4),
            "total_curtailed_kwh": round(cut, 2),
            "total_cost_usd": round(cost, 4),
            "freq_violations": fv,
            "steps_completed": n,
        },
        "feedback": f"[{gl}] Task '{task_id}' score:{sc:.3f} | Curtail:{cr:.1%} | Cost:${cost:.2f} | StabViolations:{fv}/{n} | BattHealth:{bs:.1%}",
    }

class EnergyGridEnvironment(Environment):
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Choose: {list(TASKS.keys())}")
        self._cfg = TASKS[task_id]
        self._task_id = task_id
        self._episode_id = str(uuid4())
        self._rng = random.Random(self._cfg["seed"])
        self._step_count = 0
        self._done = False
        self._history: List[Dict] = []
        self._battery_soc = 50.0
        self._battery_energy_kwh = self._cfg["battery_capacity_kwh"] * 0.5
        self._cum_cost = 0.0
        self._cum_curtailed = 0.0
        self._cum_sold = 0.0
        self._last_action: Optional[str] = None
        self._storm_events: List[Tuple[int, int, float]] = []
        self._cloud_events: List[Tuple[int, int, float]] = []
        start_h = 0 if task_id == "hard" else 6
        self._current_time = datetime(2024, 6, 15, start_h, 0)

    def reset(self, seed: Optional[int] = None) -> GridObservation:
        seed_val = seed if seed is not None else self._cfg["seed"]
        self._rng = random.Random(seed_val)
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._done = False
        self._history = []
        self._battery_soc = 50.0
        self._battery_energy_kwh = self._cfg["battery_capacity_kwh"] * 0.5
        self._cum_cost = 0.0
        self._cum_curtailed = 0.0
        self._cum_sold = 0.0
        self._last_action = None
        self._generate_weather_events()
        start_h = 0 if self._task_id == "hard" else 6
        self._current_time = datetime(2024, 6, 15, start_h, 0)
        return self._make_observation(reward=0.0)

    def step(self, action_dict) -> GridObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        action = GridAction(**action_dict) if isinstance(action_dict, dict) else action_dict
        at = action.action_type
        mag = float(action.magnitude)
        solar = self._solar()
        wind = self._wind()
        demand = self._demand()
        gen = solar + wind
        net = gen - demand
        buy_p, sell_p = self._prices()
        ts_h = self._cfg["time_step_minutes"] / 60.0
        action_rewards: Dict[str, float] = {}
        curtailed = 0.0
        sold = 0.0
        cost = 0.0
        if at == "charge_battery":
            charge_req = self._cfg["battery_max_rate_kw"] * mag
            headroom = (SOC_MAX - self._battery_soc) / 100 * self._cfg["battery_capacity_kwh"] / ts_h
            charged = min(charge_req, headroom)
            self._battery_energy_kwh = min(
                self._battery_energy_kwh + charged * ts_h * CHRG_EFF,
                self._cfg["battery_capacity_kwh"] * SOC_MAX / 100,
            )
            self._battery_soc = self._battery_energy_kwh / self._cfg["battery_capacity_kwh"] * 100
            self._battery_soc = max(0.0, min(100.0, self._battery_soc))
            if net >= 0:
                if charged > net:
                    grid_imp = charged - net
                    cost += grid_imp * ts_h * buy_p
                    curtailed = 0.0
                else:
                    curtailed = max(0.0, net - charged)
            else:
                grid_imp = abs(net) + charged
                cost += grid_imp * ts_h * buy_p
                curtailed = 0.0
                
            action_rewards["charge"] = 0.3 if charged > 0 else -0.05

        elif at == "sell_to_grid":
            if not self._cfg["grid_sell_enabled"] or self._sell_restricted():
                curtailed = max(0.0, net)
                action_rewards["restrict"] = -0.5
            else:
                sell_kw = max(0.0, net) * mag
                sold = sell_kw
                rev = sell_kw * ts_h * sell_p
                cost -= rev
                self._cum_sold += sell_kw * ts_h
                curtailed = max(0.0, net - sell_kw)
                action_rewards["revenue"] = min(rev / 10.0, 0.4)

        elif at == "curtail_power":
            curtailed = max(0.0, net) * mag

        else:  # hold
            if net > 0:
                curtailed = net * 0.5
            action_rewards["hold"] = -0.05

        if net < 0 and at != "charge_battery":
            deficit = abs(net)
            avail = min(
                self._cfg["battery_max_rate_kw"],
                max(0.0, (self._battery_soc - SOC_MIN) / 100 * self._cfg["battery_capacity_kwh"] / ts_h)
            )
            disch = min(deficit, avail)
            
            self._battery_energy_kwh = max(
                self._battery_energy_kwh - disch * ts_h / DSCH_EFF,
                self._cfg["battery_capacity_kwh"] * SOC_MIN / 100,
            )
            self._battery_soc = self._battery_energy_kwh / self._cfg["battery_capacity_kwh"] * 100
            
            remain = deficit - disch
            if remain > 0:
                imp = remain * ts_h
                cost += imp * buy_p
                action_rewards["import"] = -min(imp * buy_p / 5.0, 0.5)

        self._cum_curtailed += curtailed * ts_h
        self._cum_cost += max(0, cost)
        cpn = -(curtailed * ts_h * self._cfg["curtailment_penalty_multiplier"] * 0.05)
        fdev = self._freq_dev(curtailed, gen + demand)
        freq = FREQ_NOM + fdev
        fpn = 0.0 if abs(fdev) < 0.2 else -0.3 * abs(fdev) / 0.5
        stab = max(0.0, 1.0 - abs(fdev) / 0.5)
        soc = self._battery_soc
        brwd = 0.1 if 30 <= soc <= 80 else (0.02 if (20 <= soc < 30 or 80 < soc <= 90) else -0.15)
        erwd = -cost / 20.0
        srwd = stab * 0.4
        imbalance = abs(net)
        imbalance_penalty = -min(imbalance / max(demand, 1.0), 1.0) * 0.3
        penalty = 0.0
        if mag < 0:
            penalty -= 0.2
        if at not in VALID_ACTIONS:
            penalty -= 0.5

        raw_reward = (
            srwd + erwd + cpn + fpn + brwd +
            sum(action_rewards.values()) +
            imbalance_penalty +
            penalty
        )
        raw_reward = max(-2.0, min(2.0, raw_reward))
        total_reward = float((raw_reward + 2.0) / 4.0)
        self._last_action = at
        self._step_count += 1
        self._current_time += timedelta(minutes=self._cfg["time_step_minutes"])
        done = self._step_count >= self._cfg["episode_length_steps"]
        self._done = done
        self._history.append({
            "step": self._step_count,
            "action": at,
            "reward": total_reward,
            "solar_kw": solar,
            "wind_kw": wind,
            "demand_kw": demand,
            "battery_soc": self._battery_soc,
            "curtailed_kw": curtailed,
            "sold_kw": sold,
            "cost": cost,
            "freq_hz": freq,
        })
        return self._make_observation(reward=total_reward)

    @property
    def state(self) -> "GridState":
        return GridState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            episode_done=self._done,
            history_length=len(self._history),
            battery_soc_pct=round(self._battery_soc, 2),
            cumulative_cost=round(self._cum_cost, 4),
            cumulative_curtailed_kwh=round(self._cum_curtailed, 4),
            cumulative_sold_kwh=round(self._cum_sold, 4),
        )

    def grade(self) -> Dict[str, Any]:
        return grade_episode(self._task_id, self._history)

    def evaluate_task(self) -> float:
        if not self._history:
            return 0.0
        result = grade_episode(self._task_id, self._history)
        return float(result.get("score", 0.0))

    def _make_observation(self, reward: float = 0.0) -> GridObservation:
        solar = self._solar()
        wind = self._wind()
        demand = self._demand()
        gen = solar + wind
        net = gen - demand
        bp, sp = self._prices()
        carbon = self._carbon()
        fdev = self._freq_dev(max(0, net), gen + demand)
        stab = max(0.0, 1.0 - abs(fdev) / 0.5)
        m = self._current_time.month
        if m in [3, 4, 5, 6]:
            season = "summer"
        elif m in [7, 8, 9]:
            season = "monsoon"
        else:
            season = "winter"
        fn = 0.05 * self._cfg["volatility"]
        if season == "monsoon":
            fn *= 1.2
        return GridObservation(
            hour_of_day=round(self._current_time.hour + self._current_time.minute / 60.0, 3),
            day_of_week=self._current_time.weekday(),
            season=season,
            solar_output_kw=round(solar, 2),
            wind_output_kw=round(wind, 2),
            total_generation_kw=round(gen, 2),
            building_demand_kw=round(demand, 2),
            demand_forecast_1h_kw=round(max(0, demand * (1 + self._rng.gauss(0, fn))), 2),
            demand_forecast_3h_kw=round(max(0, demand * (1 + self._rng.gauss(0, fn * 2))), 2),
            battery_soc_pct=round(self._battery_soc, 2),
            battery_capacity_kw=self._cfg["battery_max_rate_kw"],
            battery_energy_kwh=round(self._battery_energy_kwh, 2),
            battery_max_kwh=self._cfg["battery_capacity_kwh"],
            grid_frequency_hz=round(FREQ_NOM + fdev, 4),
            grid_voltage_pu=round(VOLT_NOM + self._rng.gauss(0, 0.005), 4),
            grid_buy_price=round(bp, 4),
            grid_sell_price=round(sp, 4),
            carbon_intensity=round(carbon, 1),
            net_power_kw=round(net, 2),
            grid_stability_score=round(stab, 4),
            step=self._step_count,
            max_steps=self._cfg["episode_length_steps"],
            cumulative_cost=round(self._cum_cost, 4),
            cumulative_curtailed_kwh=round(self._cum_curtailed, 4),
            cumulative_sold_kwh=round(self._cum_sold, 4),
            last_action=self._last_action,
            done=self._done,
            reward=round(reward, 6),
            task_id=self._task_id,
        )

    def _solar(self) -> float:
        h = self._current_time.hour + self._current_time.minute / 60.0
        base = max(0.0, math.sin(math.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
        cf = 1.0
        for s, e, sev in self._cloud_events:
            if s <= self._step_count <= e:
                cf = 1.0 - sev * (0.6 + 0.4 * self._rng.random())
                break
        return max(0.0, self._cfg["solar_capacity_kw"] * base * cf *
                   (1 + self._rng.gauss(0, 0.02 * self._cfg["volatility"])))

    def _wind(self) -> float:
        if self._cfg["wind_capacity_kw"] == 0:
            return 0.0
        h = self._current_time.hour
        wm = 1.0 + 0.3 * math.cos(2 * math.pi * (h - 3) / 24)
        sf = 1.0
        for s, e, sev in self._storm_events:
            if s <= self._step_count <= e:
                sf = 1.0 + sev * 1.5
                break
        spd = max(0.0, wm * sf * (2.5 + self._rng.gauss(0, 0.5 * self._cfg["volatility"])))
        pf = 0.0 if spd < 3 or spd > 25 else ((spd - 3) / 9) ** 3 if spd < 12 else 1.0
        return self._cfg["wind_capacity_kw"] * pf

    def _demand(self) -> float:
        h = self._current_time.hour + self._current_time.minute / 60.0
        day = self._current_time.weekday()
        if h < 6:
            bf = 0.25
        elif h < 9:
            bf = 0.25 + 0.5 * (h - 6) / 3
        elif h < 18:
            bf = 0.85 - 0.1 * math.exp(-((h - 12.5) ** 2) / 2)
        elif h < 21:
            bf = 0.85 - 0.35 * (h - 18) / 3
        else:
            bf = max(0, 0.5 - 0.25 * (h - 21) / 3)
        if day >= 5:
            bf *= 0.6
        return max(0.0, self._cfg["building_max_demand_kw"] * bf *
                   (1 + self._rng.gauss(0, 0.05 * self._cfg["volatility"])))

    def _prices(self) -> Tuple[float, float]:
        h = self._current_time.hour
        if 9 <= h < 21:
            return (max(0.05, 0.25 + 0.05 * self._rng.gauss(0, self._cfg["volatility"])),
                    max(0.02, 0.18 + 0.03 * self._rng.gauss(0, self._cfg["volatility"])))
        return (max(0.05, 0.12 + 0.02 * self._rng.gauss(0, self._cfg["volatility"])),
                max(0.02, 0.08 + 0.01 * self._rng.gauss(0, self._cfg["volatility"])))

    def _carbon(self) -> float:
        return max(50.0, (300 if 9 <= self._current_time.hour < 21 else 180) +
                   self._rng.gauss(0, 30))

    def _freq_dev(self, imb: float, sys: float) -> float:
        return (imb / max(sys, 1)) * 0.3 + self._rng.gauss(0, 0.01)

    def _sell_restricted(self) -> bool:
        return any(s <= self._step_count <= e and sev > 0.7
                   for s, e, sev in self._storm_events)

    def _generate_weather_events(self):
        steps = self._cfg["episode_length_steps"]
        self._cloud_events = []
        for _ in range(max(1, int(steps * self._cfg["volatility"] * 0.1))):
            s = self._rng.randint(0, max(0, steps - 10))
            d = self._rng.randint(5, 20)
            self._cloud_events.append((s, min(s + d, steps), self._rng.uniform(0.2, 0.9)))
        self._storm_events = []
        if self._task_id == "hard":
            for _ in range(self._rng.randint(3, 6)):
                s = self._rng.randint(0, max(0, steps - 20))
                d = self._rng.randint(12, 36)
                self._storm_events.append((s, min(s + d, steps), self._rng.uniform(0.5, 0.95)))