from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ── Order ─────────────────────────────────────────────────────────────────────
class Order(BaseModel):
    id: str
    items: List[str]
    priority: int
    depends_on: Optional[str] = None   # long-horizon dependency chain


# ── Per-Robot State ────────────────────────────────────────────────────────────
class RobotState(BaseModel):
    pos: List[int]
    battery: float
    carrying: List[str]


# ── Observation (Multi-Agent) ──────────────────────────────────────────────────
class Observation(BaseModel):
    robots: Dict[str, RobotState]
    inventory: Dict[str, List[int]]
    orders: List[Order]
    goal: List[int]
    obstacles: List[List[int]]
    steps: int
    completed_orders: List[str]
    charge_station: List[int]


# ── Action ────────────────────────────────────────────────────────────────────
class Action(BaseModel):
    agent_id: str = "agent1"           # defaults for backward compat
    action_type: str                   # move | pick | drop | charge
    direction: Optional[str] = None
    item_id: Optional[str] = None


# ── Reward ────────────────────────────────────────────────────────────────────
class Reward(BaseModel):
    value: float
    breakdown: Optional[Dict[str, float]] = None


# ── Oversight Metrics (Fleet AI) ──────────────────────────────────────────────
class OversightMetrics(BaseModel):
    efficiency_score: float
    collisions: int
    idle_steps: int
    coordination_score: int
    charge_conflicts: int
    intervention_count: int
    efficiency_improvements: int
    explanation: Optional[str] = None


# ── Reset Request ─────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: str = "easy"
