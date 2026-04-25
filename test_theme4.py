from warehouse_env.env_core import WarehouseEnv
from warehouse_env.models import Action

env = WarehouseEnv()

# ── Feature 3 (Expert Feedback) ──────────────────────────────────────────────
obs = env.reset(task="easy")
_, _, _, info = env.step(Action(agent_id="agent1", action_type="move", direction="right"))
assert "expert_feedback" in info, "Missing expert_feedback"
print(f"[F3] expert_feedback='{info['expert_feedback']}'")

# Battery low → "prioritize charging earlier"
obs = env.reset(task="easy")
env.state_manager.robots["agent1"]["battery"] = 10.0
_, _, _, info = env.step(Action(agent_id="agent1", action_type="move", direction="right"))
fb = info["expert_feedback"]
print(f"[F3 low-bat] feedback='{fb}'")
assert "charg" in fb.lower(), f"Expected charging advice, got: {fb}"

# High collisions → "avoid obstacles"
obs = env.reset(task="easy")
env.state_manager.collisions = 5
_, _, _, info = env.step(Action(agent_id="agent1", action_type="move", direction="right"))
fb = info["expert_feedback"]
print(f"[F3 collisions] feedback='{fb}'")
assert "obstacle" in fb.lower(), f"Expected obstacle advice, got: {fb}"

# ── Feature 2 (Dynamic task gen) ─────────────────────────────────────────────
obs = env.reset(task="easy", difficulty=2)
sm = env.state_manager
n_items = len(sm.inventory)
n_orders = len(sm.orders)
n_obs = len(sm.obstacles)
print(f"[F2 difficulty=2] items={n_items} orders={n_orders} obstacles={n_obs} grid={sm.grid_size}")
assert n_items >= 3, f"Expected >=3 items at difficulty=2, got {n_items}"
assert n_orders >= 2, f"Expected >=2 orders at difficulty=2, got {n_orders}"

obs = env.reset(task="easy", difficulty=0)   # static (backward compat)
print(f"[F2 difficulty=0] items={len(sm.inventory)} orders={len(sm.orders)} (static, backward compat)")

# ── Feature 1 (Adaptive difficulty fields) ───────────────────────────────────
assert sm.difficulty == 0  # set by last reset
obs = env.reset(task="medium", difficulty=4)
assert sm.difficulty == 4
print(f"[F1] sm.difficulty={sm.difficulty} after reset(difficulty=4)")

# Dynamic with dep chain (difficulty>=3)
obs = env.reset(task="hard", difficulty=3)
deps = [o.get("depends_on") for o in sm.orders]
has_dep = any(d is not None for d in deps)
print(f"[F1 difficulty=3] orders with depends_on: {deps}  has_dep={has_dep}")

# ── Both agents still work post-reset ────────────────────────────────────────
obs = env.reset(task="easy", difficulty=1)
_, _, _, info = env.step(Action(agent_id="agent1", action_type="move", direction="right"))
_, _, _, info = env.step(Action(agent_id="agent2", action_type="move", direction="left"))
assert "agent_intent" in info
assert "expert_feedback" in info
assert "oversight" in info
print(f"[ALL] intent={info['agent_intent']}  feedback='{info['expert_feedback']}'")

print("\n[ALL THEME 4 FEATURES VERIFIED PASSED]")
