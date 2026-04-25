# Technical Details: Smart Warehouse Environment

This document contains the deep technical architecture, reward breakdowns, and design philosophies of the Smart Warehouse Multi-Agent Environment.

## Problem

Current LLMs and RL agents face two fundamental challenges blocking real-world deployment:

**1. Multi-Agent Coordination** -- Most benchmarks train a single agent in isolation. Real systems (warehouse fleets, logistics networks, robot swarms) require agents to cooperate, compete for shared resources, and resolve conflicts in real time, without explicit communication.

**2. Long-Horizon Planning Under Uncertainty** -- Completing a chain of dependent tasks (pick to transport to deliver to validate) across hundreds of steps, in a partially observable environment where items randomly relocate and obstacles move, requires genuine planning -- not reactive behavior.

This project provides a research-grade RL environment that forces both capabilities simultaneously.

---

## Environment Overview

Two autonomous robots (agent1, agent2) share a warehouse grid and must cooperate to fulfill a queue of orders under hard physical and logical constraints:

| Constraint | Description |
|---|---|
| Partial Observability | Each agent sees only items/obstacles within radius = 2 cells |
| Dynamic Obstacles | Hard mode regenerates obstacle positions every step |
| Stochastic World | Items randomly relocate (1% per step) -- agents must re-plan on stale beliefs |
| Order Dependencies | order3 cannot complete until order2 is done; order2 needs order1 first |
| Charge Contention | Only one agent can charge at a time -- forces resource negotiation |
| Battery Drain | Each move costs battery; battery <= 0 triggers a catastrophic -2.0 penalty |

Three task difficulties, each requiring progressively deeper planning:

| Task | Grid | Battery | Orders | Steps | Dependencies |
|------|------|---------|--------|-------|--------------|
| easy | 6x6 | 100 | 1 | 150 | none |
| medium | 8x8 | 80 | 2 | 200 | none |
| hard | 10x10 | 50 | 3 | 300 | chained (order1 -> order2 -> order3) |

---

## Key Innovations

### 1. Cooperative + Competitive Multi-Agent Dynamics

Agents simultaneously cooperate (to complete orders and earn the +1.5 cooperation bonus) and compete (for charge station access and item priority). 

**Intent Conflict Detection** -- A novel per-step signal:
- If both robots target the same item: `competition_penalty = -0.05` fires
- If robots efficiently split different items: `task_split_bonus = +0.10` fires
- `coordination_efficiency` score in [0,1] measures fleet-wide task split quality

### 2. Fleet AI -- Hierarchical Oversight Agent

The Fleet AI is a zero-action meta-agent that sits above both robots, observes all state, and intervenes when safety or efficiency rules fire. It operates as a supervisory controller -- directly analogous to a human fleet manager in logistics.

| Capability | Mechanism | Benefit |
|---|---|---|
| Emergency battery routing | Navigate to charger when battery <= 10 | Prevents -2.0 catastrophic penalty |
| Wasteful charge prevention | Redirect to delivery if battery >= 90 and carrying | +0.10 efficiency reward |
| Predictive risk assessment | Forecasts battery_risk and collision_risk one step ahead | +0.03 prevention bonus |
| Dependency violation flagging | Detects drop attempts violating order chains | Logs natural-language explanation |

Every Fleet AI decision is logged with a human-readable explanation in `info["fleet_ai_explanation"]`. The `/oversight` endpoint exposes all intervention metrics, making the system fully auditable.

### 3. Long-Horizon Planning with Subgoal Chains

Each order completion requires a 4-step subgoal sequence:
  `pick -> transport -> deliver -> validate (dependency check)`

On the hard task with 3 chained orders, agents must plan up to 12 ordered subgoals over 300 steps. A dedicated `planning_failure` penalty (-0.05) fires when an agent attempts delivery before its dependency is satisfied.

### 4. Self-Improving Adaptive Curriculum

The training environment automatically scales difficulty based on agent performance:
  `avg_reward > 0.6`  =>  `difficulty_level += 1`  (bigger grid, more obstacles, dep chains)
  `avg_reward < 0.3`  =>  `difficulty_level -= 1`  (stabilise and relearn)

At difficulty >= 1, tasks are generated dynamically -- grid size, item count, order count, and dependency depth all scale with mastery.

---

## System Architecture

```text
+----------------------------------------------------------+
|           FastAPI Server (server/app.py)                 |
|   POST /reset   POST /step   GET /state   GET /oversight |
+-----------------------------+----------------------------+
                              |
              +---------------v--------------+
              |         WarehouseEnv         |   env_core.py
              |                              |
              |   +------------------------+ |
              |   |       Fleet AI         | |  <- Zero-action oversight
              |   |  predict, intervene,   | |     Logs every decision
              |   |  explain               | |
              |   +-----------+------------+ |
              |               |              |
              |   +----------+-----------+   |
              |   |     StateManager     |   |  state_manager.py
              |   |  19-component reward |   |  Grid, robots, orders,
              |   |  Adaptive difficulty |   |  obstacles, stochastic world
              |   +-----+----------+----+    |
              +---------|-----------|--------+
                        |           |
                   [agent1]     [agent2]
               cooperative   cooperative
```

---

## Reward System -- 19 Components, Tanh-Normalized

All rewards pass through `tanh(x)` normalization to `[0,1]`, preserving gradient signal for both positive and negative events. For ease of analysis, these are grouped into four clean categories in the environment's `info` output: `safety_score`, `efficiency_score`, `cooperation_score`, and `task_progress`.

| Component | Value | Purpose |
|---|---|---|
| movement | -0.01 | Encourage step efficiency |
| collision | -0.30 | Penalise wall/obstacle navigation |
| agent_collision | -0.50 | Strong multi-agent coordination signal |
| pickup | +0.30 | Dense reward for task progress |
| delivery | +0.50/order | Primary task completion signal |
| sequencing | +0.30 | Reward correct dependency ordering |
| dependency_violation | -0.20 | Penalise out-of-order delivery |
| coordination | +0.05 | Reward agents in separate cells |
| cooperation_bonus | +1.50 | Reward true bilateral teamwork |
| competition_penalty | -0.05 | Penalise duplicate item targeting |
| task_split_bonus | +0.10 | Reward efficient task partitioning |
| completion_bonus | +1.00 | Episode success signal |
| oversight | +0.10 | Reward Fleet AI efficiency saves |
| prediction | +0.03 | Reward proactive risk prevention |
| charge_conflict | -0.10 | Penalise station resource waste |
| battery_dead | -2.00 | Catastrophic failure penalty |
| planning_failure | -0.05 | Penalise sequencing errors |
| subtask | +0.02 | Dense shaping for pick/order milestones |
| theory_of_mind | +0.05/-0.02 | Theory of Mind belief prediction reward |
