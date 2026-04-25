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


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":   "running",
        "version":  "3.0.0",
        "agents":   ["agent1", "agent2"],
        "observer": "fleet_ai",
        "tasks":    ["easy", "medium", "hard"],
        "features": [
            "multi-agent-cooperation",
            "fleet-ai-oversight",
            "order-dependencies",
            "charge-contention",
            "partial-observability",
            "curriculum-learning",
            "long-horizon-planning",
        ],
    }


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

    agent_id defaults to "agent1" for backward compatibility.
    Example: {"agent_id": "agent2", "action_type": "move", "direction": "right"}
    """
    obs, reward, done, info = env.step(action)
    return {
        "observation":      obs.model_dump(),
        "reward":           reward.value,
        "reward_breakdown": info.get("reward_breakdown", {}),
        "done":             done,
        "info":             info,
    }


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