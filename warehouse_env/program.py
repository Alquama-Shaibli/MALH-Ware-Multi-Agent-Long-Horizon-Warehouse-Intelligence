"""
warehouse_env/program.py
========================
Theme #2 — Work-Order Programs (Long-Horizon Planning)

A WorkOrderProgram wraps a sequence/DAG of timed constraints that the
agents must satisfy across an episode. The program layer sits above
the base order system and adds:

  deadline(order_id, step)     — order must be completed by `step`
  ordering(order_a, order_b)   — order_a before order_b (mirrors depends_on but at program level)
  scan_before_pick(item_id)    — agent must visit item location before picking
  escort(agent1, agent2, step) — both agents must be adjacent at `step`

Audit checkpoints fire every AUDIT_INTERVAL steps and record compliance
metrics that are exposed in info["program_progress"].
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set


AUDIT_INTERVAL = 50   # steps between checkpoints


# ── Constraint types ──────────────────────────────────────────────────────────

class Constraint:
    def __init__(self, ctype: str, **kwargs: Any) -> None:
        self.ctype  = ctype
        self.params = kwargs
        self.met    = False
        self.violated = False

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.ctype, "params": self.params,
                "met": self.met, "violated": self.violated}


# ── WorkOrderProgram ──────────────────────────────────────────────────────────

class WorkOrderProgram:
    """
    Encapsulates a set of timed constraints for one episode.
    Stateful — reset alongside the environment.
    """

    def __init__(self, program_id: str, constraints: List[Constraint]) -> None:
        self.program_id   = program_id
        self.constraints  = constraints
        self.violations:  List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.scanned_items: Set[str] = set()   # for scan_before_pick

    # ── Step-level update ──────────────────────────────────────────────────────
    def on_step(self, sm: Any, step: int) -> Dict[str, Any]:
        """
        Called every env step. Returns reward_delta and audit info.
        `sm` is the StateManager instance.
        """
        reward_delta = 0.0
        events: List[str] = []

        for c in self.constraints:
            if c.met or c.violated:
                continue

            # ── deadline ──────────────────────────────────────────────────────
            if c.ctype == "deadline":
                oid   = c.params["order_id"]
                dlstep = c.params["step"]
                if oid in sm.completed_orders:
                    c.met = True
                    reward_delta += 0.1
                    events.append(f"deadline met: {oid} by step {step}")
                elif step >= dlstep:
                    c.violated = True
                    reward_delta -= 0.1
                    self.violations.append({"step": step, "constraint": c.to_dict()})
                    events.append(f"deadline MISSED: {oid} (due step {dlstep})")

            # ── ordering (program-level, beyond base depends_on) ───────────────
            elif c.ctype == "ordering":
                a, b = c.params["order_a"], c.params["order_b"]
                if a in sm.completed_orders and b in sm.completed_orders:
                    # Check order of completion
                    idx_a = sm.completed_orders.index(a)
                    idx_b = sm.completed_orders.index(b)
                    if idx_a < idx_b:
                        c.met = True
                        reward_delta += 0.05
                    else:
                        c.violated = True
                        reward_delta -= 0.05
                        self.violations.append({"step": step, "constraint": c.to_dict()})
                        events.append(f"ordering violation: {b} completed before {a}")

            # ── scan_before_pick ───────────────────────────────────────────────
            elif c.ctype == "scan_before_pick":
                item = c.params["item_id"]
                # Mark scanned when any robot visits item location
                item_loc = sm.inventory.get(item)
                if item_loc:
                    for robot in sm.robots.values():
                        if robot["pos"] == item_loc:
                            self.scanned_items.add(item)
                # Violation check handled in on_pick_attempt — see below

            # ── escort ────────────────────────────────────────────────────────
            elif c.ctype == "escort":
                a1, a2, due = c.params["agent1"], c.params["agent2"], c.params["step"]
                if step >= due:
                    r1 = sm.robots.get(a1)
                    r2 = sm.robots.get(a2)
                    if r1 and r2:
                        dist = abs(r1["pos"][0] - r2["pos"][0]) + abs(r1["pos"][1] - r2["pos"][1])
                        if dist <= 1:
                            c.met = True
                            reward_delta += 0.05
                            events.append(f"escort met at step {step}")
                        else:
                            c.violated = True
                            reward_delta -= 0.05
                            self.violations.append({"step": step, "constraint": c.to_dict()})
                            events.append(f"escort MISSED at step {step} (dist={dist})")

        # ── Audit checkpoint ──────────────────────────────────────────────────
        audit_info: Optional[Dict[str, Any]] = None
        if step > 0 and step % AUDIT_INTERVAL == 0:
            audit_info = self._run_audit(sm, step)
            self.checkpoints.append(audit_info)

        return {
            "reward_delta":   reward_delta,
            "events":         events,
            "audit":          audit_info,
        }

    def on_pick_attempt(self, agent_id: str, item_id: str) -> float:
        """Returns penalty if scan_before_pick constraint is violated."""
        for c in self.constraints:
            if c.ctype == "scan_before_pick" and c.params["item_id"] == item_id:
                if item_id not in self.scanned_items:
                    c.violated = True
                    self.violations.append({
                        "constraint": c.to_dict(),
                        "note": f"{agent_id} picked {item_id} without scanning",
                    })
                    return -0.08
        return 0.0

    # ── Audit ─────────────────────────────────────────────────────────────────
    def _run_audit(self, sm: Any, step: int) -> Dict[str, Any]:
        total  = len(self.constraints)
        met    = sum(1 for c in self.constraints if c.met)
        viol   = sum(1 for c in self.constraints if c.violated)
        pending = total - met - viol
        score   = round(met / max(total, 1), 3)
        return {
            "step":            step,
            "compliance_score": score,
            "met":             met,
            "violated":        viol,
            "pending":         pending,
            "total":           total,
            "recoveries":      0,   # future: track constraint recovery
        }

    # ── Progress info ──────────────────────────────────────────────────────────
    def progress_info(self, sm: Any, step: int) -> Dict[str, Any]:
        total  = len(self.constraints)
        met    = sum(1 for c in self.constraints if c.met)
        viol   = sum(1 for c in self.constraints if c.violated)
        next_c = [
            c.to_dict() for c in self.constraints
            if not c.met and not c.violated
        ][:3]
        return {
            "program_id":       self.program_id,
            "compliance_score": round(met / max(total, 1), 3),
            "met":              met,
            "violated":         viol,
            "total":            total,
            "violations":       self.violations[-5:],
            "next_constraints": next_c,
            "checkpoints":      self.checkpoints[-3:],
        }

    # ── Final audit ───────────────────────────────────────────────────────────
    def final_audit(self, sm: Any, step: int) -> Dict[str, Any]:
        return self._run_audit(sm, step)


# ── Built-in programs ─────────────────────────────────────────────────────────

def get_program(program_id: str) -> Optional[WorkOrderProgram]:
    """
    Factory — returns a pre-built WorkOrderProgram by ID.
    Returns None for unknown IDs (legacy tasks use no program).
    """
    if program_id == "easy_deadline":
        return WorkOrderProgram("easy_deadline", [
            Constraint("deadline", order_id="order1", step=100),
        ])

    if program_id == "medium_ordered":
        return WorkOrderProgram("medium_ordered", [
            Constraint("deadline",  order_id="order1", step=120),
            Constraint("deadline",  order_id="order2", step=180),
            Constraint("ordering",  order_a="order1", order_b="order2"),
        ])

    if program_id == "hard_full":
        return WorkOrderProgram("hard_full", [
            Constraint("deadline",        order_id="order1", step=100),
            Constraint("deadline",        order_id="order2", step=200),
            Constraint("deadline",        order_id="order3", step=280),
            Constraint("ordering",        order_a="order1", order_b="order2"),
            Constraint("ordering",        order_a="order2", order_b="order3"),
            Constraint("scan_before_pick",item_id="item1"),
            Constraint("scan_before_pick",item_id="item2"),
            Constraint("escort",          agent1="agent1", agent2="agent2", step=150),
        ])

    if program_id == "judge":
        # Full showcase program for judge mode
        return WorkOrderProgram("judge", [
            Constraint("deadline",        order_id="order1", step=80),
            Constraint("deadline",        order_id="order2", step=160),
            Constraint("deadline",        order_id="order3", step=260),
            Constraint("ordering",        order_a="order1", order_b="order2"),
            Constraint("ordering",        order_a="order2", order_b="order3"),
            Constraint("scan_before_pick",item_id="item1"),
            Constraint("scan_before_pick",item_id="item3"),
            Constraint("escort",          agent1="agent1", agent2="agent2", step=100),
        ])

    return None
