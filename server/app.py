import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, Body
from warehouse_env.env_core import WarehouseEnv
from warehouse_env.models import Action, ResetRequest

app = FastAPI(
    title="Smart Warehouse Multi-Agent Environment + Fleet AI",
    description=(
        "OpenEnv-compatible multi-agent warehouse RL environment with Fleet AI oversight. "
        "Two cooperative agents (agent1, agent2) + one Fleet AI observer meta-agent. "
        "Features: order dependencies, charge contention, partial observability, "
        "curriculum training, and real-time intervention with explanation."
    ),
    version="3.0.0",
)

env = WarehouseEnv()
env.reset()

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = os.path.join(_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "UI not found."}

@app.get("/health")
def health_check():
    return {"status": "running", "ui": "enabled"}

@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    obs = env.reset(task=req.task)
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation":      obs.model_dump(),
        "reward":           reward.value,
        "reward_breakdown": info.get("reward_breakdown", {}),
        "done":             done,
        "info":             info,
    }

# ── Predict ────────────────────────────────────────────────────────────────────
from pydantic import BaseModel
import pickle
import numpy as np

Q_TABLE_PATH = "q_table.pkl"
q_table = {}
if os.path.exists(Q_TABLE_PATH):
    try:
        with open(Q_TABLE_PATH, "rb") as f:
            q_table = pickle.load(f)
        print(f"[Q-Table] Loaded {len(q_table)} states from {Q_TABLE_PATH}")
    except Exception as e:
        print(f"[Q-Table] WARNING: Could not load ({e}). Using heuristic fallback.")
        q_table = {}

class PredictRequest(BaseModel):
    agent_id: str

def state_to_key(state, agent_id):
    r1 = state.get("robots", {}).get(agent_id)
    if not r1:
        return (0, 0, False, False, False)
    pos = r1["pos"]
    carrying = len(r1["carrying"]) > 0
    battery = r1.get("battery", 100)
    low_battery = battery < 20
    obstacle_nearby = False
    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
        nx, ny = pos[0]+dx, pos[1]+dy
        if any(list(ob)==[nx,ny] if isinstance(ob,list) else ob.get("pos")==[nx,ny]
               for ob in state.get("obstacles", [])):
            obstacle_nearby = True
            break
    if carrying and state.get("goal"):
        target = state["goal"]
    elif state.get("inventory"):
        inv = state["inventory"]
        if isinstance(inv, dict):
            first_item = next(iter(inv.values()), None)
            target = first_item if first_item else pos
        elif isinstance(inv, list) and len(inv) > 0:
            item = inv[0]
            target = item.get("pos", pos) if isinstance(item, dict) else item
        else:
            target = pos
    else:
        target = pos
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    return (max(-1,min(1,dx)), max(-1,min(1,dy)), carrying, obstacle_nearby, low_battery)

@app.post("/predict")
def predict(req: PredictRequest):
    """Returns action from Q-table or smart heuristic with Fleet AI coordination."""
    state = env.state()
    key   = state_to_key(state, req.agent_id)

    if key in q_table:
        q_vals = q_table[key]
        best   = max(q_vals, key=q_vals.get)
        if isinstance(best, tuple):
            action_type, direction = best
        else:
            action_type, direction = "move", best
        return {"agent_id": req.agent_id, "action_type": action_type,
                "direction": direction, "q_values": {str(k):v for k,v in q_vals.items()},
                "source": "trained_policy"}

    # ── Heuristic fallback ────────────────────────────────────────────────────
    robots    = state.get("robots", {})
    robot     = robots.get(req.agent_id)
    if not robot:
        return {"agent_id": req.agent_id, "action_type": "move",
                "direction": "right", "source": "fallback"}

    pos       = robot["pos"]
    battery   = robot.get("battery", 100)
    carrying  = robot.get("carrying", [])
    inv       = state.get("inventory", {})       # {item_id: [r, c]}
    goal      = state.get("goal", [0, 0])
    charge    = state.get("charge_station", [0, 0])
    grid_size = state.get("grid_size", 6)
    fleet_ai  = env.fleet_ai
    obs_raw   = state.get("obstacles", [])

    def mdist(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def navigate(target):
        """2-pass greedy navigation:
        Pass 1: avoid walls + obstacles + other agents (preferred path).
        Pass 2: avoid walls + obstacles only (deadlock breaker)."""
        if pos == target:
            return "up"
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        if abs(dx) >= abs(dy):
            dirs = (("down" if dx>0 else "up"),
                    ("right" if dy>0 else "left"),
                    ("up"   if dx>0 else "down"),
                    ("left" if dy>0 else "right"))
        else:
            dirs = (("right" if dy>0 else "left"),
                    ("down"  if dx>0 else "up"),
                    ("left" if dy>0 else "right"),
                    ("up"   if dx>0 else "down"))
        deltas   = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
        obs_set  = {(o[0],o[1]) if isinstance(o,(list,tuple)) else (o["pos"][0],o["pos"][1])
                    for o in obs_raw}
        ag_set   = {(p[0],p[1]) for p in other_pos}
        # Pass 1: avoid obstacles AND agents
        for d in dirs:
            ddx, ddy = deltas[d]
            nx, ny   = pos[0]+ddx, pos[1]+ddy
            if not (0<=nx<grid_size and 0<=ny<grid_size): continue
            if (nx,ny) in obs_set: continue
            if (nx,ny) in ag_set:  continue
            return d
        # Pass 2: deadlock breaker — avoid only obstacles
        for d in dirs:
            ddx, ddy = deltas[d]
            nx, ny   = pos[0]+ddx, pos[1]+ddy
            if not (0<=nx<grid_size and 0<=ny<grid_size): continue
            if (nx,ny) in obs_set: continue
            return d
        return dirs[0]   # truly wall-trapped

    # Other agent context
    other_pos      = [r["pos"] for aid,r in robots.items() if aid != req.agent_id]
    other_carried  = {item for aid,r in robots.items() if aid != req.agent_id
                      for item in r.get("carrying", [])}
    # Targets the OTHER agent is ACTIVELY fetching (correct self-exclusion by agent_id key)
    other_fetching = [v["target"] for aid,v in fleet_ai.agent_intents.items()
                      if aid != req.agent_id and v.get("mode") == "fetching"]

    # ═══════════════════════════════════════════════════════════════════════════
    # P1 — AT GOAL + CARRYING → DROP  (absolute top priority, never skip)
    # ═══════════════════════════════════════════════════════════════════════════
    if carrying and pos == goal:
        fleet_ai.register_intent(req.agent_id, goal, "delivering")
        return {"agent_id": req.agent_id, "action_type": "drop",
                "direction": None, "source": "heuristic_fallback",
                "fleet_ai_coord": "Dropping at goal"}

    # ═══════════════════════════════════════════════════════════════════════════
    # P2 — LOW BATTERY → CHARGE
    # Threshold: enough steps to reach charger + 5 safety margin (min 20)
    # ═══════════════════════════════════════════════════════════════════════════
    dist_to_charge   = mdist(pos, charge)
    charge_threshold = max(25, dist_to_charge + 5)
    if battery <= charge_threshold:
        fleet_ai.register_intent(req.agent_id, charge, "charging")
        if pos == charge:
            return {"agent_id": req.agent_id, "action_type": "charge",
                    "direction": None, "source": "heuristic_fallback",
                    "fleet_ai_coord": None}
        return {"agent_id": req.agent_id, "action_type": "move",
                "direction": navigate(charge), "source": "heuristic_fallback",
                "fleet_ai_coord": f"Charging: batt={int(battery)}, dist={dist_to_charge}"}

    # ═══════════════════════════════════════════════════════════════════════════
    # P3 — NOT CARRYING → FETCH nearest assigned item
    # ═══════════════════════════════════════════════════════════════════════════
    if not carrying:
        unclaimed = {iid: loc for iid, loc in inv.items() if iid not in other_carried}

        # Standing ON an unclaimed item → PICK immediately
        for iid, loc in unclaimed.items():
            if loc == pos:
                fleet_ai.register_intent(req.agent_id, pos, "fetching")
                return {"agent_id": req.agent_id, "action_type": "pick",
                        "direction": None, "source": "heuristic_fallback",
                        "fleet_ai_coord": None}

        if unclaimed:
            # Deterministic split — agent1 takes even indices (0,2,4…),
            # agent2 takes odd indices (1,3,5…). Zero competition by design.
            agent_idx    = 0 if req.agent_id == "agent1" else 1
            sorted_items = sorted(unclaimed.items(), key=lambda x: x[0])  # sort by item name
            my_items     = [(iid, loc) for i, (iid, loc) in enumerate(sorted_items)
                            if i % 2 == agent_idx]

            if not my_items:
                # My parity items all delivered → help with remaining,
                # but avoid what the other agent is already heading for
                my_items = [(iid, loc) for iid, loc in sorted_items
                            if loc not in other_fetching]
                if not my_items:
                    my_items = sorted_items   # only 1 item left, both go, first wins

            target = min(my_items, key=lambda x: mdist(pos, x[1]))[1]
            fleet_ai.register_intent(req.agent_id, target, "fetching")
            return {"agent_id": req.agent_id, "action_type": "move",
                    "direction": navigate(target), "source": "heuristic_fallback",
                    "fleet_ai_coord": f"Fleet AI: {req.agent_id} assigned item"}

        # No unclaimed items → park at charger (agent1) or near goal (agent2)
        # Agents wait on OPPOSITE sides of the grid so they never overlap
        if req.agent_id == "agent1":
            idle_spot = charge
        else:
            # agent2 waits one cell left of goal
            idle_spot = [max(0, goal[0] - 1), goal[1]]
        fleet_ai.register_intent(req.agent_id, idle_spot, "idle")
        if pos == idle_spot:
            # At idle spot: charge if agent1 at charger, else hold still (move up → wall bounce)
            if req.agent_id == "agent1":
                return {"agent_id": req.agent_id, "action_type": "charge",
                        "direction": None, "source": "heuristic_fallback",
                        "fleet_ai_coord": "Idle: charging"}
            return {"agent_id": req.agent_id, "action_type": "move",
                    "direction": "up", "source": "heuristic_fallback",
                    "fleet_ai_coord": "Idle: waiting near goal"}
        return {"agent_id": req.agent_id, "action_type": "move",
                "direction": navigate(idle_spot), "source": "heuristic_fallback",
                "fleet_ai_coord": "Parking (no items)"}

    # ═══════════════════════════════════════════════════════════════════════════
    # P4 — CARRYING → DELIVER (queue at adjacent cell if goal is blocked)
    # ═══════════════════════════════════════════════════════════════════════════
    goal_blocked = any(r["pos"] == goal for aid, r in robots.items() if aid != req.agent_id)
    if goal_blocked:
        obs_set = {(o[0],o[1]) if isinstance(o,(list,tuple)) else (o["pos"][0],o["pos"][1])
                   for o in obs_raw}
        adj  = [[goal[0]-1, goal[1]], [goal[0]+1, goal[1]],
                [goal[0],   goal[1]-1], [goal[0],   goal[1]+1]]
        free = [c for c in adj
                if 0<=c[0]<grid_size and 0<=c[1]<grid_size
                and (c[0],c[1]) not in obs_set]
        queue_spot = min(free, key=lambda c: mdist(pos, c)) if free else goal
        fleet_ai.register_intent(req.agent_id, queue_spot, "queuing")
        if pos == queue_spot:   # already at queue spot — hold still
            return {"agent_id": req.agent_id, "action_type": "move",
                    "direction": "up", "source": "heuristic_fallback",
                    "fleet_ai_coord": f"Fleet AI: {req.agent_id} queuing"}
        return {"agent_id": req.agent_id, "action_type": "move",
                "direction": navigate(queue_spot), "source": "heuristic_fallback",
                "fleet_ai_coord": f"Fleet AI: {req.agent_id} queuing — goal occupied"}

    # Goal is free → drive straight to it
    fleet_ai.register_intent(req.agent_id, goal, "delivering")
    return {"agent_id": req.agent_id, "action_type": "move",
            "direction": navigate(goal), "source": "heuristic_fallback",
            "fleet_ai_coord": None}


# ── State ──────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    """Full environment state for the UI."""
    raw_state = env.state()
    sm        = env.state_manager
    all_held  = {item for r in raw_state.get("robots", {}).values() for item in r.get("carrying", [])}
    unclaimed = [{"x": loc[0], "y": loc[1]}
                 for iid, loc in raw_state.get("inventory", {}).items()
                 if iid not in all_held]
    return {
        "grid_size":    raw_state.get("grid_size", 6),
        "agents": {
            aid: {"x": r["pos"][0], "y": r["pos"][1],
                  "battery": r["battery"], "carrying": len(r["carrying"]) > 0}
            for aid, r in raw_state.get("robots", {}).items()
        },
        "items":        unclaimed,
        "obstacles":    [{"x": o[0], "y": o[1]} for o in raw_state.get("obstacles", [])],
        "goal":         {"x": raw_state["goal"][0], "y": raw_state["goal"][1]}
                        if raw_state.get("goal") else None,
        "charge_station": {"x": raw_state["charge_station"][0], "y": raw_state["charge_station"][1]}
                          if raw_state.get("charge_station") else None,
        "deliveries":   len(raw_state.get("completed_orders", [])),
        "total_orders": len(raw_state.get("orders", [])),
        "collisions":   getattr(sm, "collisions", 0) + getattr(sm, "agent_collisions", 0),
    }


# ── Fleet AI Oversight ─────────────────────────────────────────────────────────
@app.get("/oversight")
def oversight():
    fa = env.fleet_ai
    sm = env.state_manager
    return {
        "intervention_count":      fa.intervention_count,
        "efficiency_improvements": fa.efficiency_improvements,
        "last_explanation":        fa._last_explanation or None,
        "metrics":                 fa.compute_oversight_metrics(sm),
    }


# ── OpenEnv Entry Point ────────────────────────────────────────────────────────
def main() -> FastAPI:
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)