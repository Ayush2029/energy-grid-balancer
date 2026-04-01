"""
inference.py — High-Performance Baseline Agent for Energy Grid Balancer (OpenEnv)

Targets: easy ~0.985 | medium ~0.870 | hard ~0.750
Physics ceilings (hard limits from simulation physics):
    easy   = 0.986  (battery fills past 90% grader threshold without strict cap)
    medium = 0.878  (3h evening deficit drains battery below 20% — unavoidable)
    hard   = 0.750  (3-day storm, undersized battery, severe night demand)

Architecture:
    RULE-BASED CORE  —  deterministic optimal policy from grader formula analysis
    LLM OVERRIDE     —  consulted only for genuine edge cases (storm ambiguity, end-game)
    EPISODE TRACKER  —  running metrics drive real-time tactical adjustments

Required env vars:
    API_BASE_URL   LLM API endpoint  (e.g. https://api.openai.com/v1)
    MODEL_NAME     Model identifier  (e.g. gpt-4o-mini)
    HF_TOKEN       API key

Optional:
    ENV_BASE_URL   Server URL  (default: http://localhost:7860)

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=sk-...
    python inference.py
"""

import json
import os
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN (or OPENAI_API_KEY) environment variable.")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ═══════════════════════════════════════════════════════════════════════════════
# GRADER FORMULAS (hardcoded from environment.py — exact)
# ═══════════════════════════════════════════════════════════════════════════════
# curtailment_score  = max(0, 1 - (curtailed_kwh / total_gen_kwh) * 2)
# cost_efficiency    = max(0, min(1, 1 - total_cost_usd / baseline))
#                      baseline = steps * 0.25 * (10/60) * max_demand * 0.3
# grid_stability     = 1 - (steps_where |freq-50| > 0.2Hz) / total_steps
# battery_health     = steps_where(20% <= SoC <= 90%) / total_steps
# completion         = steps_done / max_steps

WEIGHTS = {
    "easy":   {"curtail": 0.30, "cost": 0.35, "stab": 0.15, "batt": 0.10, "comp": 0.10},
    "medium": {"curtail": 0.25, "cost": 0.25, "stab": 0.25, "batt": 0.15, "comp": 0.10},
    "hard":   {"curtail": 0.20, "cost": 0.20, "stab": 0.35, "batt": 0.15, "comp": 0.10},
}

# ── Physics constants ──────────────────────────────────────────────────────────
SOC_GRADER_MAX = 90.0   # grader: above this = penalty every step
SOC_GRADER_MIN = 20.0   # grader: below this = penalty every step
SOC_OP_CEIL    = 82.0   # operational ceiling (8% below grader threshold = safety margin)
SOC_OP_FLOOR   = 22.0   # operational floor   (2% above grader threshold = safety margin)
DEMAND_MAX     = {"easy": 60.0, "medium": 140.0, "hard": 200.0}
TS_H           = 10 / 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE STATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
class EpisodeTracker:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.w = WEIGHTS[task_id]
        self.history = []           # rolling 10-step window
        self.freq_violations = 0
        self.bad_soc_steps = 0
        self.total_gen_kwh = 0.0
        self.consecutive_low_stab = 0
        self.suspected_storm = False

    def update(self, obs: dict, action: dict):
        soc  = obs.get("battery_soc_pct", 50)
        freq = obs.get("grid_frequency_hz", 50.0)
        stab = obs.get("grid_stability_score", 1.0)
        gen  = obs.get("total_generation_kw", 0)
        self.total_gen_kwh += gen * TS_H
        if abs(freq - 50.0) > 0.2:
            self.freq_violations += 1
        if soc < SOC_GRADER_MIN or soc > SOC_GRADER_MAX:
            self.bad_soc_steps += 1
        # Storm detection: two consecutive steps with stability < 0.5
        if stab < 0.5:
            self.consecutive_low_stab += 1
            if self.consecutive_low_stab >= 2:
                self.suspected_storm = True
        else:
            self.consecutive_low_stab = max(0, self.consecutive_low_stab - 1)
            if self.consecutive_low_stab == 0:
                self.suspected_storm = False
        self.history.append({"soc": soc, "stab": stab,
                              "net": obs.get("net_power_kw", 0),
                              "action": action.get("action_type", "hold")})
        if len(self.history) > 10:
            self.history.pop(0)

    def soc_trend(self) -> str:
        if len(self.history) < 3:
            return "unknown"
        delta = self.history[-1]["soc"] - self.history[-3]["soc"]
        return "rising" if delta > 3 else "falling" if delta < -3 else "stable"

    def live_score_estimate(self, obs: dict) -> float:
        step = obs.get("step", 1)
        if step == 0:
            return 0.5
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL RULE-BASED POLICY
# Derived from exhaustive grader analysis.
# ═══════════════════════════════════════════════════════════════════════════════
def rule_action(obs: dict, tracker: EpisodeTracker) -> dict:
    soc   = obs.get("battery_soc_pct",      50.0)
    net   = obs.get("net_power_kw",          0.0)
    hour  = obs.get("hour_of_day",          12.0) % 24
    stab  = obs.get("grid_stability_score",  1.0)
    sp    = obs.get("grid_sell_price",       0.15)
    step  = obs.get("step",                  0)
    maxst = obs.get("max_steps",             48)
    batt_cap_kw  = obs.get("battery_capacity_kw",  50.0)
    batt_max_kwh = obs.get("battery_max_kwh",     100.0)

    task = tracker.task_id
    storm = tracker.suspected_storm or stab < 0.4

    # Contextual flags
    is_peak  = 9  <= hour <= 21
    pre_eve  = 14 <= hour <= 18      # charge window before nightfall
    end_game = step >= maxst * 0.92  # last 8% — sell any remaining SoC

    # Dynamic charge ceiling — higher before night to buffer the demand surge
    charge_to = 84.0 if (pre_eve and task in ("medium", "hard")) else SOC_OP_CEIL

    # Per-step charge magnitude: how much of batt rate capacity to use
    charge_mag = round(min(1.0, max(0.1, net / max(batt_cap_kw, 1))), 2)

    # ── SURPLUS (generation > demand) ─────────────────────────────────────────
    if net > 1.0:

        # End-game: sell everything possible (no point saving for next episode)
        if end_game and soc > 30:
            if storm:
                return {"action_type": "curtail_power", "magnitude": 0.5}
            return {"action_type": "sell_to_grid", "magnitude": 0.95}

        # Battery below ceiling → CHARGE
        if soc < charge_to:
            return {"action_type": "charge_battery", "magnitude": charge_mag}

        # Battery at/above ceiling → SELL (or curtail if storm)
        if not storm:
            sell_mag = 0.95 if is_peak else 0.80
            return {"action_type": "sell_to_grid", "magnitude": sell_mag}

        # Storm + battery full → minimum curtailment
        return {"action_type": "curtail_power", "magnitude": 0.5}

    # ── DEFICIT / BALANCED (demand >= generation) ─────────────────────────────
    # Auto-discharge handles it. HOLD is always correct here.
    return {"action_type": "hold", "magnitude": 0.5}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """\
You are an expert energy grid AI agent. You control a real-time renewable energy microgrid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADER FORMULAS (EXACT — memorise these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
battery_health  = steps_where(20% ≤ SoC ≤ 90%) / total_steps
                  ABOVE 90% = penalty every step. Use 82% as operational cap.
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
OPTIMAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SoC MUST stay in [22%, 82%] every step. This is non-negotiable.

SURPLUS (net > 1kW):
  SoC < 82%  →  charge_battery  (magnitude = min(1.0, net/battery_rate_kw))
  SoC ≥ 82%, no storm  →  sell_to_grid  (0.95 peak, 0.80 off-peak)
  SoC ≥ 82%, storm     →  curtail_power  (0.5)

DEFICIT (net ≤ 1kW):
  hold  (auto-discharge covers it; never sell during deficit)

STORM (stability_score < 0.4 or two consecutive < 0.5):
  NEVER sell_to_grid (causes -0.5 reward penalty from restriction)
  Charge any surplus, curtail if battery full

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


def get_action(obs: dict, tracker: EpisodeTracker) -> dict:
    """
    Hybrid agent:
    1. Compute deterministic rule action (always optimal for normal cases)
    2. Detect genuine edge cases → query LLM for resolution
    3. Fall back to rule if LLM fails or edge case not detected
    """
    rule = rule_action(obs, tracker)
    soc   = obs.get("battery_soc_pct", 50)
    net   = obs.get("net_power_kw", 0)
    stab  = obs.get("grid_stability_score", 1.0)
    step  = obs.get("step", 0)
    maxst = obs.get("max_steps", 48)
    hour  = obs.get("hour_of_day", 12) % 24
    est   = tracker.live_score_estimate(obs)

    # Edge case 1: Storm ambiguity (stability in grey zone, not clearly storm)
    edge = None
    if 0.4 <= stab <= 0.65 and net > 5 and abs(soc - SOC_OP_CEIL) < 6:
        edge = (f"Stab={stab:.3f} is ambiguous (grey zone 0.4-0.65). "
                f"Storm={tracker.suspected_storm}. SoC={soc:.1f}% NET={net:+.1f}kW. "
                f"Is sell safe or will it be restricted?")

    # Edge case 2: End-game with significant stored energy
    elif step >= int(maxst * 0.95) and soc > 60 and net >= 0:
        edge = (f"End-game (step {step}/{maxst}). SoC={soc:.1f}%, net={net:+.1f}kW. "
                f"Maximise sell revenue in remaining {maxst-step} steps.")

    # Edge case 3: Score recovery — we're below target and need tactical shift
    elif (est < 0.80 and step > maxst * 0.40
          and tracker.task_id in ("easy", "medium")
          and tracker.bad_soc_steps > step * 0.15):
        edge = (f"Score {est:.3f} < 0.80 at {step}/{maxst}. "
                f"BadSoC={tracker.bad_soc_steps}/{step} steps. "
                f"What's the highest-impact adjustment?")

    if edge:
        llm = llm_action(obs, tracker, edge)
        if llm:
            return llm

    return rule


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def wait_for_server(max_retries: int = 20, delay: float = 3.0):
    for attempt in range(1, max_retries + 1):
        try:
            if requests.get(f"{ENV_BASE_URL}/health", timeout=5).status_code == 200:
                print(f"  ✓ Server ready at {ENV_BASE_URL}")
                return
        except requests.RequestException:
            pass
        print(f"  Waiting for server... ({attempt}/{max_retries})")
        time.sleep(delay)
    print(f"\nERROR: Server at {ENV_BASE_URL} did not respond.")
    sys.exit(1)


def run_task(task_id: str) -> dict:
    print(f"\n{'═'*62}\n  TASK: {task_id.upper()}\n{'═'*62}")
    print(f"[START] task={task_id}")
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": 42},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
    sid  = data["session_id"]
    obs  = data["observation"]
    max_steps = obs["max_steps"]
    print(f"  Session  : {sid[:24]}...")
    print(f"  Task     : {data['info']['task_name']}")
    print(f"  Steps    : {max_steps}")

    tracker = EpisodeTracker(task_id)
    total_r = 0.0; step = 0; done = False
    llm_calls = 0
    log_every = max(1, max_steps // 8)

    while not done and step < max_steps:
        action = get_action(obs, tracker)

        sr = requests.post(f"{ENV_BASE_URL}/step",
                           json={"session_id": sid, "action": action}, timeout=15)
        if sr.status_code != 200:
            print(f"  Step {step} HTTP {sr.status_code}: {sr.text[:80]}")
            break

        sd = sr.json()
        obs = sd["observation"]; done = sd["done"]
        total_r += sd["reward"]["total"]; step += 1
        tracker.update(obs, action)
        print(f"[STEP] task={task_id} step={step} reward={sd['reward']['total']:.4f} action={action['action_type']}")

        if step % log_every == 0 or done:
            est = tracker.live_score_estimate(obs)
            print(f"  [{step:4d}/{max_steps}]  "
                  f"NET={obs['net_power_kw']:+6.1f}kW  "
                  f"SoC={obs['battery_soc_pct']:5.1f}%  "
                  f"act={action['action_type']:<16s}  "
                  f"freq={obs['grid_frequency_hz']:.3f}Hz  "
                  f"cost=${obs['cumulative_cost']:.2f}  "
                  f"est≈{est:.3f}")
        time.sleep(0.01)

    gr = requests.post(f"{ENV_BASE_URL}/grade", json={"session_id": sid}, timeout=15)
    gr.raise_for_status()
    gd = gr.json()
    score = gd.get("score", 0.0)
    bd = gd.get("breakdown", {})

    print(f"\n  ── Final Score : {score:.4f}")
    print(f"  ── Feedback    : {gd.get('feedback','')}")
    for k, v in bd.items():
        warn = " ⚠" if (k=="battery_health_score" and float(v)<0.85) else ""
        warn = warn or (" ⚠" if (k=="cost_efficiency_score" and float(v)<0.75) else "")
        print(f"       {k:<34s}: {v}{warn}")

    print(f"[END] task={task_id} score={score:.4f}")
    return {"task_id": task_id, "score": score, "breakdown": bd,
            "total_reward": round(total_r, 4), "steps_completed": step,
            "llm_calls": llm_calls, "feedback": gd.get("feedback", "")}


def main():
    print(f"\n{'═'*62}")
    print(f"  ⚡  ENERGY GRID BALANCER — BASELINE INFERENCE")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL}")
    print(f"  Env URL : {ENV_BASE_URL}")
    print(f"{'═'*62}\n")

    wait_for_server()

    results = []
    for task_id in ["easy", "medium", "hard"]:
        try:
            results.append(run_task(task_id))
        except Exception as e:
            print(f"\n  ERROR on task '{task_id}': {e}")
            results.append({"task_id": task_id, "score": 0.0, "error": str(e), "breakdown": {}})

    print(f"\n{'═'*62}\n  FINAL SCORES\n {'═'*62}")
    for r in results:
        s   = r.get("score", 0.0)
        bar = "█" * int(s * 26) + "░" * (26 - int(s * 26))
        err = "  ⚠ error" if "error" in r else ""
        print(f"  {r['task_id']:8s}  [{bar}]  {s:.4f}{err}")

    valid = [r["score"] for r in results if "error" not in r]
    avg   = sum(valid) / len(valid) if valid else 0.0
    print(f"\n  AVERAGE  : {avg:.4f}  ({len(valid)}/3 tasks)\n{'═'*62}\n")

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "api_base_url": API_BASE_URL,
            "agent": "hybrid",
            "results": results, "average_score": round(avg, 4),
            "tasks_completed": len(valid),
        }, f, indent=2)
    print("  Saved → baseline_results.json")


if __name__ == "__main__":
    main()
