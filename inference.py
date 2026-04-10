import json
import os
import sys
import time
from typing import Optional, List
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = os.environ.get("BENCHMARK",    "energy-grid-balancer")
if not HF_TOKEN:
    sys.stderr.write("ERROR: Set HF_TOKEN (or OPENAI_API_KEY) environment variable.\n")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
MAX_LLM_CALLS = 30
WEIGHTS = {
    "easy":   {"curtail": 0.30, "cost": 0.35, "stab": 0.15, "batt": 0.10, "comp": 0.10},
    "medium": {"curtail": 0.25, "cost": 0.25, "stab": 0.25, "batt": 0.15, "comp": 0.10},
    "hard":   {"curtail": 0.20, "cost": 0.20, "stab": 0.35, "batt": 0.15, "comp": 0.10},
}
SOC_GRADER_MAX = 90.0   
SOC_GRADER_MIN = 20.0   
SOC_OP_CEIL = {
    "easy":   88.0,
    "medium": 88.0,
    "hard":   88.0,
}
SOC_OP_FLOOR = {
    "easy":   22.0,   
    "medium": 28.0,   
    "hard":   36.0,   
}
DEMAND_MAX     = {"easy": 60.0, "medium": 140.0, "hard": 200.0}
TS_H           = 10 / 60.0
MIN_SELL_PRICE = {
    "easy":   0.10,   
    "medium": 0.12,   
    "hard":   0.14,   
}

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

class EpisodeTracker:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.w = WEIGHTS[task_id]
        self.history = []           
        self.freq_violations = 0
        self.bad_soc_steps = 0
        self.total_gen_kwh = 0.0
        self.consecutive_low_stab = 0
        self.suspected_storm = False
        self.storm_lockout = 0
        self.last_sold_kwh = 0.0
        self.last_action_type = "hold"
        self.last_net = 0.0  
        self.llm_calls = 0
        self.steps_since_llm = 999  

    def update(self, obs: dict, action: dict):
        soc  = obs.get("battery_soc_pct", 50)
        freq = obs.get("grid_frequency_hz", 50.0)
        stab = obs.get("grid_stability_score", 1.0)
        gen  = obs.get("total_generation_kw", 0)
        self.total_gen_kwh += gen * TS_H
        self.steps_since_llm += 1
        if abs(freq - 50.0) > 0.2:
            self.freq_violations += 1
        if soc < SOC_GRADER_MIN or soc > SOC_GRADER_MAX:
            self.bad_soc_steps += 1
            
        if stab < 0.5:
            self.consecutive_low_stab += 1
            if self.consecutive_low_stab >= 2:
                self.suspected_storm = True
        else:
            self.consecutive_low_stab = max(0, self.consecutive_low_stab - 1)
            if self.consecutive_low_stab == 0:
                self.suspected_storm = False

        sold_now = obs.get("cumulative_sold_kwh", 0)
        if self.last_action_type == "sell_to_grid" and self.last_net > 1.0:
            if abs(sold_now - self.last_sold_kwh) < 0.01:
                self.storm_lockout = 37 
                
        self.last_sold_kwh = sold_now
        self.last_action_type = action.get("action_type", "hold")
        self.last_net = obs.get("net_power_kw", 0)
        
        if self.storm_lockout > 0:
            self.storm_lockout -= 1

        self.history.append({"soc": soc, "stab": stab,
                              "net": obs.get("net_power_kw", 0),
                              "action": action.get("action_type", "hold")})
        if len(self.history) > 10:
            self.history.pop(0)

    def soc_trend(self) -> str:
        if len(self.history) < 3: return "unknown"
        delta = self.history[-1]["soc"] - self.history[-3]["soc"]
        return "rising" if delta > 3 else "falling" if delta < -3 else "stable"

    def battery_health_ratio(self, step: int) -> float:
        if step == 0: return 1.0
        return (step - self.bad_soc_steps) / step

    def live_score_estimate(self, obs: dict) -> float:
        step = obs.get("step", 1)
        if step == 0: return 0.5
        batt_ok = (step - self.bad_soc_steps) / step
        stab_ok = (step - self.freq_violations) / step
        curtailed = obs.get("cumulative_curtailed_kwh", 0)
        gen = max(self.total_gen_kwh, 0.001)
        curt_score = max(0.0, 1.0 - (curtailed / gen) * 2)
        cost = obs.get("cumulative_cost", 0)
        max_steps = obs.get("max_steps", 48)
        baseline = max_steps * 0.25 * TS_H * DEMAND_MAX.get(self.task_id, 100) * 0.3
        cost_score = max(0.0, min(1.0, 1.0 - cost / max(baseline, 0.01)))
        w = self.w
        return (w["curtail"] * curt_score + w["cost"] * cost_score +
                w["stab"] * stab_ok + w["batt"] * batt_ok + w["comp"] * 1.0)

def rule_action(obs: dict, tracker: EpisodeTracker) -> dict:
    soc          = obs.get("battery_soc_pct",        50.0)
    net          = obs.get("net_power_kw",             0.0)
    hour         = obs.get("hour_of_day",             12.0) % 24
    stab         = obs.get("grid_stability_score",     1.0)
    step         = obs.get("step",                       0)
    maxst        = obs.get("max_steps",                 48)
    batt_rate    = obs.get("battery_capacity_kw",       50.0)
    task  = tracker.task_id
    floor = SOC_OP_FLOOR[task]
    ceil  = SOC_OP_CEIL[task]
    storm = tracker.storm_lockout > 0 or (task == "hard" and stab < 0.35)
    is_pre_peak = (6 <= hour < 9)     
    is_night = (21 <= hour < 24) or (0 <= hour < 6) 
    end_game = (step >= maxst * 0.92 and soc > 30)

    if end_game and net > 1.0 and not storm:
        return {"action_type": "sell_to_grid", "magnitude": 1.0}

    if is_pre_peak:
        if task == "hard" and soc < floor + 30:
            return {"action_type": "charge_battery", "magnitude": 1.0}
        elif task == "medium" and soc < floor + 20: 
            return {"action_type": "charge_battery", "magnitude": 1.0}
    elif is_night:
        if task == "hard" and soc < floor + 15:
            return {"action_type": "charge_battery", "magnitude": 1.0}
        elif task == "medium" and soc < floor + 5: 
            return {"action_type": "charge_battery", "magnitude": 1.0}

    if net > 1.0:
        if storm:
            if soc < ceil:
                return {"action_type": "charge_battery", "magnitude": 1.0}
            return {"action_type": "curtail_power", "magnitude": 1.0}

        if soc < ceil:
            if net > batt_rate: 
                return {"action_type": "sell_to_grid", "magnitude": 1.0}
            return {"action_type": "charge_battery", "magnitude": 1.0}

        return {"action_type": "sell_to_grid", "magnitude": 1.0}

    return {"action_type": "hold", "magnitude": 0.5}

SYSTEM_PROMPT = """\
You are an expert energy grid AI agent. You control a real-time renewable energy microgrid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADER FORMULAS (EXACT — memorise these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
battery_health  = steps_where(20% ≤ SoC ≤ 90%) / total_steps
                  ABOVE 90% = penalty every step. Use task ceil as operational cap.
curtailment     = max(0, 1 - (curtailed_kwh / gen_kwh) * 2)
cost_efficiency = max(0, min(1, 1 - total_cost / baseline))
                  Sell revenue SUBTRACTS from cost → sell aggressively at peak
grid_stability  = 1 - (freq_violation_steps / total_steps)
                  violation = |freq - 50Hz| > 0.2Hz

WEIGHTS:
  EASY   → Cost 35% | Curtailment 30% | Stability 15% | Battery 10% | Completion 10%
  MEDIUM → Cost 25% | Curtailment 25% | Stability 25% | Battery 15% | Completion 10%
  HARD   → Stability 35% | Cost 20%   | Curtailment 20% | Battery 15% | Completion 10%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMAL RULES (task-aware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SoC floors: EASY=22%, MEDIUM=28%, HARD=36%  
SoC ceilings: EASY/MEDIUM/HARD=88%

SURPLUS (net > 1kW):
  SoC < task_ceil  →  charge_battery (magnitude = min(1.0, net/battery_rate_kw))
  SoC ≥ task_ceil, no storm, sell_price ≥ threshold  →  sell_to_grid (scale with price)
  SoC ≥ task_ceil, storm or low price  →  curtail_power (use 1.0 magnitude to prevent imbalance)

DEFICIT (net ≤ 1kW):
  hold (battery auto-discharge handles deficit efficiently)

STORM (stability_score < 0.4 or two consecutive < 0.5):
  NEVER sell_to_grid (causes -0.5 reward penalty)
  Charge any surplus, curtail 100% if battery is full and grid is unstable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT: valid JSON only, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"action_type": "charge_battery|sell_to_grid|curtail_power|hold", "magnitude": 0.0-1.0, "reasoning": "one line"}
"""

def llm_action(obs: dict, tracker: EpisodeTracker, edge_context: str) -> Optional[dict]:
    soc  = obs.get("battery_soc_pct", 50)
    net  = obs.get("net_power_kw",     0)
    hour = obs.get("hour_of_day",     12) % 24
    step = obs.get("step",             0)
    maxst= obs.get("max_steps",       48)
    est  = tracker.live_score_estimate(obs)
    prompt = (
        f"Task={tracker.task_id.upper()} | Step {step}/{maxst} | Hour={hour:.1f}h\n"
        f"Solar={obs.get('solar_output_kw',0):.1f}kW  Wind={obs.get('wind_output_kw',0):.1f}kW  "
        f"Demand={obs.get('building_demand_kw',0):.1f}kW  NET={net:+.1f}kW\n"
        f"SoC={soc:.1f}% ({obs.get('battery_energy_kwh',0):.0f}/{obs.get('battery_max_kwh',100):.0f}kWh) "
        f"Trend={tracker.soc_trend()}\n"
        f"Freq={obs.get('grid_frequency_hz',50):.3f}Hz  Stab={obs.get('grid_stability_score',1):.3f}  "
        f"Storm={tracker.suspected_storm}\n"
        f"Buy=${obs.get('grid_buy_price',0.2):.3f}  Sell=${obs.get('grid_sell_price',0.15):.3f}  "
        f"Peak={'YES' if 9<=hour<=21 else 'NO'}\n"
        f"Forecast1h={obs.get('demand_forecast_1h_kw',0):.1f}kW  "
        f"Forecast3h={obs.get('demand_forecast_3h_kw',0):.1f}kW\n"
        f"CumCost=${obs.get('cumulative_cost',0):.2f}  "
        f"Curtailed={obs.get('cumulative_curtailed_kwh',0):.1f}kWh  "
        f"Sold={obs.get('cumulative_sold_kwh',0):.1f}kWh\n"
        f"LiveScore≈{est:.3f}  FreqViol={tracker.freq_violations}  BadSoC={tracker.bad_soc_steps}\n"
        f"EDGE CASE: {edge_context}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=100, temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        for fence in ["```json", "```"]:
            if fence in txt:
                txt = txt.split(fence)[1].split("```")[0].strip()
                break
        d = json.loads(txt)
        return {"action_type": d.get("action_type","hold"),
                "magnitude": max(0.0, min(1.0, float(d.get("magnitude", 0.5))))}
    except Exception:
        return None

def clamp_action(action: dict) -> dict:
    return {
        "action_type": action.get("action_type", "hold"),
        "magnitude": max(0.0, min(1.0, float(action.get("magnitude", 0.5))))
    }

def get_action(obs: dict, tracker: EpisodeTracker) -> tuple[dict, bool]:
    rule = rule_action(obs, tracker)
    soc   = obs.get("battery_soc_pct", 50)
    net   = obs.get("net_power_kw", 0)
    stab  = obs.get("grid_stability_score", 1.0)
    step  = obs.get("step", 0)
    maxst = obs.get("max_steps", 48)
    floor = SOC_OP_FLOOR[tracker.task_id]
    edge = None
    if tracker.steps_since_llm >= 10: 
        if 0.4 <= stab <= 0.65 and net > 5 and abs(soc - SOC_OP_CEIL[tracker.task_id]) < 6:
            edge = (f"Stab={stab:.3f} is ambiguous (grey zone 0.4-0.65). "
                    f"Storm={tracker.suspected_storm}. SoC={soc:.1f}% NET={net:+.1f}kW. "
                    f"Is sell safe or will it be restricted?")
        elif step >= int(maxst * 0.95) and soc > 60 and net >= 0:
            edge = (f"End-game (step {step}/{maxst}). SoC={soc:.1f}%, net={net:+.1f}kW. "
                    f"Maximise sell revenue in remaining {maxst-step} steps.")
        elif (tracker.task_id == "hard" and soc < floor + 6 and step < maxst * 0.90):
            edge = (f"SoC={soc:.1f}% is dangerously close to floor={floor}% on hard task. "
                    f"Step {step}/{maxst}. Net={net:+.1f}kW. Avoid grader penalty.")
        elif tracker.task_id != "easy" and step > 0 and step % 60 == 0:
            edge = "Periodic strategic check to improve global score"

    if edge and tracker.llm_calls < MAX_LLM_CALLS:
        llm = llm_action(obs, tracker, edge)
        if llm:
            tracker.steps_since_llm = 0 
            return clamp_action(llm), True

    return clamp_action(rule), False

def wait_for_server(max_retries: int = 20, delay: float = 3.0):
    for attempt in range(1, max_retries + 1):
        try:
            if requests.get(f"{ENV_BASE_URL}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(delay)
    sys.stderr.write(f"ERROR: Server at {ENV_BASE_URL} did not respond.\n")
    sys.exit(1)

def run_task(task_id: str) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30
        )
        r.raise_for_status()
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        log_end(success=False, steps=0, score=0.00, rewards=[0.00])
        return {"task_id": task_id, "score": 0.0, "error": str(e)}
        
    data = r.json()
    sid  = data["session_id"]
    obs  = data["observation"]
    tracker = EpisodeTracker(task_id)
    step = 0
    done = False
    episode_rewards = []
    success = False
    score = 0.0
    try:
        while True:
            action, used_llm = get_action(obs, tracker)
            if used_llm:
                tracker.llm_calls += 1

            sr = requests.post(f"{ENV_BASE_URL}/step",
                               json={"session_id": sid, "action": action}, timeout=15)
            if sr.status_code != 200:
                log_step(step=step+1, action="error", reward=0.00, done=True, error=f"HTTP_{sr.status_code}")
                break

            sd = sr.json()
            obs = sd["observation"]
            step_reward = float(sd["reward"]["total"])
            done = sd["done"]
            tracker.update(obs, action)
            episode_rewards.append(step_reward)
            action_str = f"{action['action_type']}({action['magnitude']})"
            log_step(step=step+1, action=action_str, reward=step_reward, done=done, error=None)
            step += 1
            if done: 
                break

        gr = requests.post(f"{ENV_BASE_URL}/grade", json={"session_id": sid}, timeout=15)
        gr.raise_for_status()
        gd = gr.json()
        score = float(gd.get("score", 0.0))
        success = score > 0.0  
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        log_step(step=step+1, action="error", reward=0.00, done=True, error=error_msg)

    log_end(success=success, steps=step, score=score, rewards=episode_rewards)
    return {
        "task_id": task_id, 
        "score": score, 
        "steps_completed": step,
        "llm_calls": tracker.llm_calls
    }

def main():
    wait_for_server()
    results = []
    for task_id in ["easy", "medium", "hard"]:
        try:
            results.append(run_task(task_id))
        except Exception as e:
            results.append({"task_id": task_id, "score": 0.0, "error": str(e)})

    valid = [r["score"] for r in results if "error" not in r]
    avg   = sum(valid) / len(valid) if valid else 0.0
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, 
            "api_base_url": API_BASE_URL,
            "agent": "hybrid",
            "results": results, 
            "average_score": round(avg, 4),
            "tasks_completed": len(valid),
        }, f, indent=2)

if __name__ == "__main__":
    main()