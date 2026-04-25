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

class PredictRequest(BaseModel):
    agent_id: str

@app.post("/predict")
def predict(req: PredictRequest):
    """Returns the true optimal action from the saved Q-table."""
    from online_rl import SimplePolicy
    
    # Load trained Q-table if it exists
    q_table = {}
    q_path = os.path.join(_ROOT, "q_table.pkl")
    if os.path.exists(q_path):
        with open(q_path, "rb") as f:
            q_table = pickle.load(f)
            
    policy = SimplePolicy(req.agent_id)
    policy.epsilon = 0.0  # Force pure exploitation (no random moves)
    policy.q_values = q_table
    
    obs = env._make_obs()
    action = policy.choose_action(obs)
    return action.model_dump()


# ── State ─────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    """Return full current environment state (all agents, orders, obstacles)."""
    return env.state()


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