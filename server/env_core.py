"""
server/env_core.py — Compatibility shim
========================================
Some older scripts attempt: from server.env_core import WarehouseEnv
This file re-exports everything from the canonical warehouse_env.env_core
so those imports never break.
"""
from warehouse_env.env_core import (  # noqa: F401
    WarehouseEnv,
    FleetAI,
    _STEP_LIMITS,
    _normalize,
    _detect_intent_conflict,
    _compute_coordination_efficiency,
    _expert_feedback,
    _infer_intent,
)

__all__ = [
    "WarehouseEnv",
    "FleetAI",
    "_STEP_LIMITS",
    "_normalize",
    "_detect_intent_conflict",
    "_compute_coordination_efficiency",
    "_expert_feedback",
    "_infer_intent",
]
