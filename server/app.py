import os
import sys

# Ensure project root is on sys.path when this module is imported by uvicorn
# from an unexpected working directory (e.g. HuggingFace Spaces, Docker).
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

# Module-level singleton — shared across all requests (stateful environment)
env = WarehouseEnv()
env.reset()  # Ensure initial state is populated


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure static directory exists
STATIC_DIR = os.path.join(_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static directory for JS/CSS assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── UI Visualizer Dashboard ───────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    """Serves the 2D HTML5 Canvas Visualizer Dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "UI not found. Please ensure static/index.html exists."}

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "running", "ui": "enabled"}


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset(req: ResetRequest = Body(default=ResetRequest())):
    """
    Reset to initial state.
    Optional body: {"task": "easy" | "medium" | "hard"}
    Backward-compatible: no body → easy task.
    """
    obs = env.reset(task=req.task)
    return obs.model_dump()


# ── Step ──────────────────────────────────────────────────────────────────────
@app.post("/step")
def step(action: Action):
    """
    Execute one action for one agent.
    Fleet AI may override the action silently before execution.
    """
    obs, reward, done, info = env.step(action)
    return {
        "observation":      obs.model_dump(),
        "reward":           reward.value,
        "reward_breakdown": info.get("reward_breakdown", {}),
        "done":             done,
        "info":             info,
    }


# ── Predict (Trained RL Policy) ───────────────────────────────────────────────
from pydantic import BaseModel
import pickle
import numpy as np

Q_TABLE_PATH = "q_table.pkl"

# Load Q-table (safe load — handles missing or LFS-pointer-corrupted files)
q_table = {}
if os.path.exists(Q_TABLE_PATH):
    try:
        with open(Q_TABLE_PATH, "rb") as f:
            q_table = pickle.load(f)
        print(f"[Q-Table] Loaded {len(q_table)} states from {Q_TABLE_PATH}")
    except Exception as e:
        print(f"[Q-Table] WARNING: Could not load {Q_TABLE_PATH} ({e}). Using heuristic fallback.")
        q_table = {}

class PredictRequest(BaseModel):
    agent_id: str

def state_to_key(state, agent_id):
    """Mirror of online_rl.py SimplePolicy._get_state() — must stay in sync."""
    r1 = state.get("robots", {}).get(agent_id)
    if not r1:
        return (0, 0, False, False, False)

    pos = r1["pos"]
    carrying = len(r1["carrying"]) > 0
    battery = r1.get("battery", 100)
    low_battery = battery < 20

    obstacle_nearby = False
    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
        nx, ny = pos[0]+dx, pos[1]+dy
        if any(list(ob) == [nx, ny] if isinstance(ob, list) else ob.get("pos") == [nx, ny]
               for ob in state.get("obstacles", [])):
            obstacle_nearby = True
            break

    # Determine target position
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
    return (max(-1, min(1, dx)), max(-1, min(1, dy)), carrying, obstacle_nearby, low_battery)

@app.post("/predict")
def predict(req: PredictRequest):
    """Returns the true optimal action from Q-table or smart heuristic.
    Also registers agent intent with Fleet AI for proactive coordination.
    """
    state = env.state()
    key = state_to_key(state, req.agent_id)

    if key in q_table:
        q_vals = q_table[key]
        best_action = max(q_vals, key=q_vals.get)

        # best_action is a tuple like ("move", "right") or ("pick", None)
        if isinstance(best_action, tuple):
            action_type = best_action[0]
            direction = best_action[1]
        else:
            action_type = "move"
            direction = best_action

        return {
            "agent_id": req.agent_id,
            "action_type": action_type,
            "direction": direction,
            "q_values": {str(k): v for k, v in q_vals.items()},
            "source": "trained_policy"
        }

    # ── Smart heuristic: navigate → pickup → deliver ──────────────────────────
    robots = state.get("robots", {})
    robot  = robots.get(req.agent_id)
    if robot:
        pos           = robot["pos"]
        battery       = robot.get("battery", 100)
        carrying      = robot.get("carrying", [])
        inv           = state.get("inventory", {})   # {item_id: [row, col]}
        goal          = state.get("goal", [0, 0])
        charge        = state.get("charge_station", [0, 0])
        grid_size     = state.get("grid_size", 6)
        fleet_ai      = env.fleet_ai
        obs_raw       = state.get("obstacles", [])

        def dist(a, b):   return abs(a[0]-b[0]) + abs(a[1]-b[1])
        def navigate(target):
            """Greedy navigation: return best direction toward target avoiding walls/agents."""
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            # Priority: move along longer axis first
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
            deltas = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
            obs_set = {(o[0],o[1]) if isinstance(o,(list,tuple)) else (o["pos"][0],o["pos"][1])
                       for o in obs_raw}
            for d in dirs:
                ddx,ddy = deltas[d]
                nx,ny   = pos[0]+ddx, pos[1]+ddy
                if not (0<=nx<grid_size and 0<=ny<grid_size): continue
                if (nx,ny) in obs_set:                         continue
                if [nx,ny] in other_pos:                       continue
                return d
            return dirs[0]  # fallback: take first direction even if blocked

        other_pos     = [r["pos"] for aid,r in robots.items() if aid != req.agent_id]
        other_carried = {item for aid,r in robots.items() if aid != req.agent_id
                         for item in r.get("carrying",[])}
        fetching_tgts = [v["target"] for v in fleet_ai.agent_intents.values()
                         if v.get("mode") == "fetching"
                         and list(v.values())[0] != req.agent_id]  # exclude self
        coord_msg = None

        # ══ PRIORITY 1: AT GOAL + CARRYING → DROP ════════════════════════════
        # ALWAYS drop first — never skip for battery or anything else
        if carrying and pos == goal:
            fleet_ai.register_intent(req.agent_id, goal, "delivering")
            return {"agent_id": req.agent_id, "action_type": "drop",
                    "direction": None, "source": "heuristic_fallback",
                    "fleet_ai_coord": "Dropping at goal"}

        # ══ PRIORITY 2: CRITICAL BATTERY → CHARGE ════════════════════════════
        if battery < 15:
            fleet_ai.register_intent(req.agent_id, charge, "charging")
            if pos == charge:
                return {"agent_id": req.agent_id, "action_type": "charge",
                        "direction": None, "source": "heuristic_fallback",
                        "fleet_ai_coord": None}
            return {"agent_id": req.agent_id, "action_type": "move",
                    "direction": navigate(charge), "source": "heuristic_fallback",
                    "fleet_ai_coord": "Emergency charge"}

        # ══ PRIORITY 3: NOT CARRYING → FETCH ══════════════════════════════════
        if not carrying:
            unclaimed = {iid: loc for iid, loc in inv.items() if iid not in other_carried}

            # Standing ON an unclaimed item → pick immediately
            for iid, loc in unclaimed.items():
                if loc == pos:
                    fleet_ai.register_intent(req.agent_id, pos, "fetching")
                    return {"agent_id": req.agent_id, "action_type": "pick",
                            "direction": None, "source": "heuristic_fallback",
                            "fleet_ai_coord": None}

            if unclaimed:
                # ── DETERMINISTIC SPLIT: No competition possible ──────────
                # Sort items by ID (stable). Agent1 takes indices 0,2,4...
                # Agent2 takes indices 1,3,5... → they never target same item
                agent_idx     = 0 if req.agent_id == "agent1" else 1
                sorted_items  = sorted(unclaimed.items(), key=lambda x: x[0])
                my_items      = {iid: loc for i,(iid,loc) in enumerate(sorted_items)
                                 if i % 2 == agent_idx}

                if not my_items:
                    # All "my" items delivered — take from other set (help finish)
                    my_items = dict(sorted_items)

                target = min(my_items.values(), key=lambda l: dist(pos, l))
                fleet_ai.register_intent(req.agent_id, target, "fetching")
                return {"agent_id": req.agent_id, "action_type": "move",
                        "direction": navigate(target), "source": "heuristic_fallback",
                        "fleet_ai_coord": f"Fleet AI: {req.agent_id} → assigned item"}

            # No unclaimed items → retreat to charger and wait
            fleet_ai.register_intent(req.agent_id, charge, "idle")
            return {"agent_id": req.agent_id, "action_type": "move",
                    "direction": navigate(charge), "source": "heuristic_fallback",
                    "fleet_ai_coord": None}


        # ══ PRIORITY 4: CARRYING → DELIVER (queue if goal blocked) ═══════════
        goal_blocked = any(r["pos"] == goal for aid,r in robots.items() if aid != req.agent_id)
        if goal_blocked:
            # Find nearest free cell adjacent to goal → queue there
            obs_set = {(o[0],o[1]) if isinstance(o,(list,tuple)) else (o["pos"][0],o["pos"][1])
                       for o in obs_raw}
            adj  = [[goal[0]-1,goal[1]], [goal[0]+1,goal[1]],
                    [goal[0],goal[1]-1], [goal[0],goal[1]+1]]
            free = [c for c in adj
                    if 0<=c[0]<grid_size and 0<=c[1]<grid_size
                    and c not in other_pos and (c[0],c[1]) not in obs_set]
            target   = min(free, key=lambda c: dist(pos,c)) if free else goal
            coord_msg = f"Fleet AI: {req.agent_id} queuing — goal occupied"
            fleet_ai.register_intent(req.agent_id, target, "queuing")
            # If already at queue spot, just wait (stay still)
            if pos == target:
                return {"agent_id": req.agent_id, "action_type": "move",
                        "direction": "up", "source": "heuristic_fallback",
                        "fleet_ai_coord": coord_msg}
        else:
            target = goal
            fleet_ai.register_intent(req.agent_id, target, "delivering")

        return {"agent_id": req.agent_id, "action_type": "move",
                "direction": navigate(target), "source": "heuristic_fallback",
        }

    return {"agent_id": req.agent_id, "action_type": "move", "direction": "right", "source": "fallback"}



# ── State ─────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    """Return full current environment state (all agents, orders, obstacles) tailored for the UI."""
    raw_state = env.state()
    sm = env.state_manager

    # Only show unclaimed items (not currently being carried by any agent)
    all_held = {item for r in raw_state.get("robots", {}).values() for item in r.get("carrying", [])}
    unclaimed_items = [
        {"x": loc[0], "y": loc[1]}
        for item_id, loc in raw_state.get("inventory", {}).items()
        if item_id not in all_held
    ]

    return {
        "grid_size": raw_state.get("grid_size", 10),
        "agents": {
            aid: {
                "x": r["pos"][0],
                "y": r["pos"][1],
                "battery": r["battery"],
                "carrying": len(r["carrying"]) > 0
            } for aid, r in raw_state.get("robots", {}).items()
        },
        "items": unclaimed_items,
        "obstacles": [
            {"x": obs[0], "y": obs[1]} for obs in raw_state.get("obstacles", [])
        ],
        "goal": {"x": raw_state["goal"][0], "y": raw_state["goal"][1]} if raw_state.get("goal") else None,
        "charge_station": {"x": raw_state["charge_station"][0], "y": raw_state["charge_station"][1]} if raw_state.get("charge_station") else None,
        "deliveries":     len(raw_state.get("completed_orders", [])),
        "total_orders":   len(raw_state.get("orders", [])),
        "collisions":     getattr(sm, "collisions", 0) + getattr(sm, "agent_collisions", 0)
    }


# ── Fleet AI Oversight Dashboard ──────────────────────────────────────────────
@app.get("/oversight")
def oversight():
    """
    Real-time Fleet AI oversight metrics.
    Shows: interventions, efficiency score, idle steps, charge conflicts.
    """
    fa = env.fleet_ai
    sm = env.state_manager
    return {
        "intervention_count":      fa.intervention_count,
        "efficiency_improvements": fa.efficiency_improvements,
        "last_explanation":        fa._last_explanation or None,
        "metrics":                 fa.compute_oversight_metrics(sm),
    }


# ── OpenEnv Entry Point ───────────────────────────────────────────────────────
def main() -> FastAPI:
    """
    Required by OpenEnv — callable entrypoint that returns the FastAPI app.
    openenv.yaml points to server.app:main.
    """
    return app


# ── Local Dev ─────────────────────────────────────────────────────────────────
# Note: HuggingFace Spaces uses root app.py on port 7860.
# Local development uses port 8000 here.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)