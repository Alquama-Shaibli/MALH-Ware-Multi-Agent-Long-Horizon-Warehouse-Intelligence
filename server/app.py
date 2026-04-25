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

# Load Q-table
q_table = {}
if os.path.exists(Q_TABLE_PATH):
    with open(Q_TABLE_PATH, "rb") as f:
        q_table = pickle.load(f)

class PredictRequest(BaseModel):
    agent_id: str

def state_to_key(state, agent_id):
    # Recreate the exact state representation used in online_rl.py
    r1 = state.get("robots", {}).get(agent_id)
    if not r1:
        return (0, 0, False, False)
        
    pos = r1["pos"]
    carrying = len(r1["carrying"]) > 0

    obstacle_nearby = False
    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
        nx, ny = pos[0]+dx, pos[1]+dy
        if any(ob["pos"] == [nx, ny] for ob in state.get("obstacles", [])):
            obstacle_nearby = True
            break

    if not carrying and state.get("inventory"):
        target = state["inventory"][0]["pos"]
    elif state.get("goal"):
        target = state["goal"]
    else:
        return (0, 0, carrying, obstacle_nearby)
        
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    return (int(np.sign(dx)), int(np.sign(dy)), carrying, obstacle_nearby)

@app.post("/predict")
def predict(req: PredictRequest):
    """Returns the true optimal action and q_values from the saved Q-table."""
    state = env.state()
    key = state_to_key(state, req.agent_id)

    if key in q_table:
        q_vals = q_table[key]
        best_action = max(q_vals, key=q_vals.get)
        
        # Convert tuple action (e.g., ("move", "right")) to string direction
        direction = best_action[1] if isinstance(best_action, tuple) else best_action
        
        return {
            "agent_id": req.agent_id,
            "action_type": "move",
            "direction": direction,
            "q_values": {str(k): v for k, v in q_vals.items()},
            "source": "trained_policy"
        }

    return {
        "agent_id": req.agent_id,
        "action_type": "move",
        "direction": "right",
        "source": "fallback"
    }


# ── State ─────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    """Return full current environment state (all agents, orders, obstacles) tailored for the UI."""
    raw_state = env.state()
    sm = env.state_manager
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
        "items": [
            {"x": loc[0], "y": loc[1]} for loc in raw_state.get("inventory", {}).values()
        ],
        "obstacles": [
            {"x": obs[0], "y": obs[1]} for obs in raw_state.get("obstacles", [])
        ],
        "deliveries": len(raw_state.get("completed_orders", [])),
        "collisions": getattr(sm, "collisions", 0) + getattr(sm, "agent_collisions", 0)
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