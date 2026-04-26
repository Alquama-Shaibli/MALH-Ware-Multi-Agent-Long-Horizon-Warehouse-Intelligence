import math
from typing import Any, Dict, Optional, Tuple

from warehouse_env.state_manager import StateManager
from warehouse_env.models import Action, Observation, Order, Reward, RobotState


# ── Normalization ──────────────────────────────────────────────────────────────
def _normalize(raw: float) -> float:
    """
    Tanh normalization: (-inf, +inf) → (0, 1).
    Preserves gradient signal for negative rewards — a penalty maps to ~0.43,
    not collapsed to 0.0 — so agents can distinguish bad from neutral.
    """
    return (math.tanh(raw) + 1.0) / 2.0


# ── Step limits — Theme #2: Long-horizon planning ────────────────────────────
_STEP_LIMITS = {"easy": 150, "medium": 200, "hard": 300}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Fleet AI Observer Agent
# ─────────────────────────────────────────────────────────────────────────────
class FleetAI:
    """
    The Fleet AI is a third, non-acting meta-agent that monitors both robots
    and overrides their submitted actions when safety or efficiency rules fire.

    Rules (priority order):
      1. Critical battery: force charge if battery <= 10
      2. Navigate to charge if battery <= 10 and not at station
      3. Wasteful charge: redirect to delivery if battery >= 90 and carrying
      4. Dependency violation: flag (don't block) and log
    """

    def __init__(self) -> None:
        self.intervention_count: int = 0
        self.efficiency_improvements: int = 0
        self._last_explanation: str = ""
        self._last_intervention_type: str = ""
        # Feature 1: Predictive tracking
        self._last_prediction: Dict[str, Any] = {}
        self._prediction_used: bool = False
        # Shared intent registry — agents announce their target before acting
        # Format: {agent_id: {"target": [x,y], "mode": "fetching"|"delivering"|"charging"}}
        self.agent_intents: Dict[str, Dict[str, Any]] = {}

    def register_intent(self, agent_id: str, target: list, mode: str) -> None:
        """Called by each agent before its step to announce its current plan."""
        self.agent_intents[agent_id] = {"target": target, "mode": mode}

    def reset(self) -> None:
        self.intervention_count      = 0
        self.efficiency_improvements = 0
        self._last_explanation       = ""
        self._last_intervention_type = ""
        self._last_prediction        = {}
        self._prediction_used        = False
        self.agent_intents           = {}  # Clear shared intent registry on episode reset

    def intervene(
        self,
        agent_id: str,
        intended_action: Action,
        sm: StateManager,
    ) -> Tuple[Action, bool, str]:
        """
        Evaluate the intended action. May return an override Action.
        Returns: (final_action, was_overridden, explanation_string)
        """
        robot = sm.robots.get(agent_id)
        if robot is None:
            self._last_prediction  = {}
            self._prediction_used  = False
            return intended_action, False, ""

        battery = robot["battery"]
        pos     = list(robot["pos"])
        cx, cy  = sm.charge_station
        gx, gy  = sm.goal
        carrying = robot.get("carrying", [])

        # ── Feature 1: Predictive risk assessment BEFORE intervention ─────
        prediction = self._predict_risks(agent_id, robot, sm)
        self._last_prediction = prediction
        self._prediction_used = False  # set True below if prediction guides decision

        # ── Rule 0: Idle redirect — if agent idle 3+ steps and items remain ─
        idle = sm.idle_steps.get(agent_id, 0)
        if (
            idle >= 3
            and intended_action.action_type == "move"
            and not carrying
        ):
            # Find nearest unclaimed item
            unclaimed = {
                item: loc for item, loc in sm.inventory.items()
                if not any(item in r["carrying"] for r in sm.robots.values())
            }
            if unclaimed:
                best_item, best_loc = min(
                    unclaimed.items(),
                    key=lambda kv: abs(kv[1][0] - pos[0]) + abs(kv[1][1] - pos[1]),
                )
                tx, ty = best_loc
                dx, dy = tx - pos[0], ty - pos[1]
                direction = (("down" if dx > 0 else "up") if abs(dx) >= abs(dy)
                             else ("right" if dy > 0 else "left"))
                override    = Action(agent_id=agent_id, action_type="move", direction=direction)
                explanation = (
                    f"{agent_id} idle_redirect toward {best_item} (idle={idle}) "
                    f"— FleetAI efficiency redirect."
                )
                self.intervention_count      += 1
                self.efficiency_improvements += 1
                self._last_explanation        = explanation
                self._last_intervention_type  = "idle_redirect"
                return override, True, explanation

        # ── Rule 1: At charge station with critical battery ────────────────
        if battery <= 25 and intended_action.action_type != "charge":
            if pos == [cx, cy]:
                override     = Action(agent_id=agent_id, action_type="charge")
                explanation  = (
                    f"{agent_id} forced to CHARGE (battery={battery:.0f}) "
                    f"— FleetAI prevented battery-death penalty."
                )
                self.intervention_count     += 1
                self.efficiency_improvements += 1
                self._last_explanation       = explanation
                self._last_intervention_type = "battery_save"
                self._prediction_used        = prediction["battery_risk"]
                return override, True, explanation

            # ── Rule 2: Navigate to charge station (critical battery) ──────
            dx, dy = cx - pos[0], cy - pos[1]
            direction = (("down" if dx > 0 else "up") if abs(dx) >= abs(dy)
                         else ("right" if dy > 0 else "left"))
            override    = Action(agent_id=agent_id, action_type="move", direction=direction)
            explanation = (
                f"{agent_id} rerouted toward charge station (battery={battery:.0f}) "
                f"— FleetAI emergency routing."
            )
            self.intervention_count     += 1
            self.efficiency_improvements += 1
            self._last_explanation       = explanation
            self._last_intervention_type = "battery_save"
            self._prediction_used        = prediction["battery_risk"]
            return override, True, explanation

        # ── Rule 3: Wasteful charge when battery is high ───────────────────
        if (
            intended_action.action_type == "charge"
            and battery >= 90
            and carrying
            and sm.charging_agent is None
        ):
            dx, dy = gx - pos[0], gy - pos[1]
            direction = (("down" if dx > 0 else "up") if abs(dx) >= abs(dy)
                         else ("right" if dy > 0 else "left"))
            override    = Action(agent_id=agent_id, action_type="move", direction=direction)
            explanation = (
                f"{agent_id} redirected to deliver (battery={battery:.0f}, "
                f"carrying={carrying}) — FleetAI efficiency optimization."
            )
            self.intervention_count      += 1
            self._last_explanation        = explanation
            self._last_intervention_type  = "efficiency_redirect"
            return override, True, explanation

        # ── Rule 4: Dependency violation flag ─────────────────────────────
        if intended_action.action_type == "drop" and pos == [gx, gy]:
            for order in sm.orders:
                oid = order["id"]
                if oid in sm.completed_orders:
                    continue
                dep = order.get("depends_on")
                if dep and dep not in sm.completed_orders:
                    if any(item in carrying for item in order["items"]):
                        explanation = (
                            f"{agent_id} attempted {oid} but dependency '{dep}' not met "
                            f"— FleetAI flagged sequencing violation."
                        )
                        self.intervention_count     += 1
                        self._last_explanation       = explanation
                        self._last_intervention_type = "dependency_flag"
                        return intended_action, True, explanation  # log only, don't block

        # ── Rule 5: Collision avoidance ────────────────────────────────────
        if intended_action.action_type == "move" and intended_action.direction:
            dx_map = {"up": -1, "down": 1, "left": 0, "right": 0}
            dy_map = {"up": 0,  "down": 0, "left": -1, "right": 1}
            d  = intended_action.direction
            nx = pos[0] + dx_map.get(d, 0)
            ny = pos[1] + dy_map.get(d, 0)

            for aid2, r2 in sm.robots.items():
                if aid2 == agent_id or r2["pos"] != [nx, ny]:
                    continue

                # ── Exception A: carrying agent heading to GOAL — NEVER block ──
                # The drop zone must always be reachable for agents with items.
                if carrying and [nx, ny] == [gx, gy]:
                    # Instead, force the blocking idle agent to move away next step
                    # (logged as a coordination event, not a collision)
                    explanation = (
                        f"{agent_id} delivery_priority: goal occupied by {aid2}, "
                        f"allowing entry — FleetAI delivery priority."
                    )
                    self._last_explanation       = explanation
                    self._last_intervention_type = "delivery_priority"
                    return intended_action, False, explanation  # let it through

                # ── Exception B: both agents heading same direction (convoy) ──
                # Don't redirect if the other agent is also moving away next step
                other_intent = self.agent_intents.get(aid2, {})
                if other_intent.get("mode") == "idle" and [nx, ny] == [gx, gy]:
                    # Blocking agent is idle AT the goal — force it away, let carrier through
                    explanation = (
                        f"{agent_id} delivery_priority: {aid2} idle at goal, "
                        f"allowing carrier to enter — FleetAI coordination."
                    )
                    self._last_explanation       = explanation
                    self._last_intervention_type = "delivery_priority"
                    return intended_action, False, explanation

                # ── Standard redirect ─────────────────────────────────────────
                alt_dir     = "down" if d in ("left", "right") else "right"
                override    = Action(agent_id=agent_id, action_type="move", direction=alt_dir)
                explanation = (
                    f"{agent_id} collision_avoid: redirected {d}->{alt_dir} "
                    f"(would hit {aid2}) — FleetAI collision prevention."
                )
                self.intervention_count     += 1
                self._last_explanation       = explanation
                self._last_intervention_type = "collision_avoid"
                return override, True, explanation

        self._last_explanation       = ""
        self._last_intervention_type = ""
        self._prediction_used        = False
        return intended_action, False, ""

    # ── Feature 1: Predictive Risk Assessment ─────────────────────────────────
    def _predict_risks(self, agent_id: str, robot: dict, sm: StateManager) -> Dict[str, Any]:
        """
        Scan the agent's immediate surroundings and battery level to
        predict risks BEFORE intervention logic fires.

        Returns:
            battery_risk  : bool — battery will be critical within ~5 steps
            collision_risk: bool — an obstacle or wall is within 1 cell
        """
        battery = robot["battery"]
        px, py  = robot["pos"]
        g       = sm.grid_size

        # Battery risk: critically low or will die within ~5 steps
        battery_risk = battery <= 15

        # Collision risk: any obstacle exactly 1 Manhattan step away
        adjacent = [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)]
        collision_risk = any(
            (not (0 <= nx < g and 0 <= ny < g))  # wall
            or (nx, ny) in sm.obstacles           # obstacle
            for nx, ny in adjacent
        )

        return {"battery_risk": battery_risk, "collision_risk": collision_risk}

    def compute_oversight_metrics(
        self, sm: StateManager,
        intent_analysis: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build the `oversight` sub-dict exposed in every /step info payload."""
        n_completed = len(sm.completed_orders)
        efficiency  = round(sm.steps / max(n_completed, 1), 3)
        total_idle  = sum(sm.idle_steps.values())
        metrics: Dict[str, Any] = {
            "efficiency_score":        efficiency,
            "collisions":              sm.collisions + sm.agent_collisions,
            "idle_steps":              total_idle,
            "coordination_score":      sm.coordination_events,
            "charge_conflicts":        sm.charge_conflicts,
            "intervention_count":      self.intervention_count,
            "efficiency_improvements": self.efficiency_improvements,
            "explanation":             self._last_explanation or None,
            # Feature 1: Predictive signals
            "prediction":              self._last_prediction,
            "prediction_used":         self._prediction_used,
        }
        # Feature 2: Intent analysis
        if intent_analysis is not None:
            metrics["intent_analysis"] = intent_analysis
        return metrics


# ── Theme #1: Intent Conflict Detection ──────────────────────────────────────
def _detect_intent_conflict(sm: StateManager) -> tuple:
    """
    Detect when both agents are targeting the same unclaimed item.
    Returns: (has_conflict: bool, conflict_item: str or None, efficiency_bonus: bool)
    """
    all_held = {item for r in sm.robots.values() for item in r["carrying"]}

    # Build each agent's nearest target item
    agent_targets: Dict[str, str] = {}
    for aid, robot in sm.robots.items():
        rx, ry = robot["pos"]
        best_item = None
        best_dist = float("inf")
        for item, loc in sm.inventory.items():
            if item in all_held:
                continue
            dist = abs(loc[0] - rx) + abs(loc[1] - ry)
            if dist < best_dist:
                best_dist = dist
                best_item = item
        if best_item:
            agent_targets[aid] = best_item

    targets = list(agent_targets.values())
    if len(targets) == 2 and targets[0] == targets[1]:
        return True, targets[0], False     # conflict: same item
    if len(targets) == 2 and targets[0] != targets[1]:
        return False, None, True           # efficient split
    return False, None, False


# ── Theme #1: Coordination Efficiency ────────────────────────────────────────
def _compute_coordination_efficiency(sm: StateManager) -> float:
    """
    Score in [0, 1] reflecting how well agents avoid duplicate effort.
    Based on: task split ratio, low collision rate, completed orders.
    """
    n_orders = max(len(sm.orders), 1)
    n_done   = len(sm.completed_orders)
    # Contributors diversity: reward if different agents completed different orders
    diverse = len(sm.order_contributors)
    score = (n_done / n_orders) * 0.6 + min(diverse / n_orders, 1.0) * 0.4
    # Penalise collisions
    total_col = sm.collisions + sm.agent_collisions
    score = max(0.0, score - total_col * 0.02)
    return round(min(score, 1.0), 4)


# ── Feature 3: Simulated Expert Feedback (Snorkel AI) ─────────────────────
def _expert_feedback(agent_id: str, sm: StateManager) -> str:
    """
    Rule-based expert feedback for the acting agent, generated after each step.
    Mirrors a Snorkel-style labeling function: deterministic rules over state.
    """
    robot    = sm.robots.get(agent_id, {})
    battery  = robot.get("battery", 100)
    carrying = robot.get("carrying", [])

    total_collisions = sm.collisions + sm.agent_collisions

    if total_collisions > 3:
        return "Agent should avoid obstacles more carefully"
    if battery < 15:
        return "Agent should prioritize charging earlier"
    if len(carrying) > 2:
        return "Agent should deliver items sooner"
    if len(sm.completed_orders) == len(sm.orders) and sm.orders:
        return "Excellent — all orders completed!"
    return "Good performance"


# ── Feature 2: Agent Intent Tagging ──────────────────────────────────────────
def _infer_intent(agent_id: str, sm: StateManager) -> str:
    """
    Classify the agent's current behavioural intent from its state.

    Labels (in priority order):
      seeking_charge  — battery is low, agent needs to charge soon
      delivering      — carrying items and heading toward goal
      fetching_item   — moving toward a pending, unclaimed inventory item
      exploring       — no clear sub-goal (idle or wandering)
    """
    robot = sm.robots.get(agent_id)
    if robot is None:
        return "exploring"

    battery  = robot["battery"]
    pos      = robot["pos"]
    carrying = robot["carrying"]
    gx, gy   = sm.goal
    cx, cy   = sm.charge_station

    # 1. Seeking charge
    if battery <= 20:
        return "seeking_charge"

    # 2. Delivering (carrying something and not at charge station)
    if carrying and pos != [cx, cy]:
        return "delivering"

    # 3. Fetching item — determine if any needed item is reachable
    pending_items: set = set()
    for order in sm.orders:
        if order["id"] in sm.completed_orders:
            continue
        dep = order.get("depends_on")
        if dep and dep not in sm.completed_orders:
            continue
        for item in order["items"]:
            pending_items.add(item)

    all_held = [item for r in sm.robots.values() for item in r["carrying"]]
    has_target = any(
        item not in all_held and item in pending_items
        for item in sm.inventory
    )
    if has_target:
        return "fetching_item"

    return "exploring"


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade 2: Theory of Mind — belief inference
# ─────────────────────────────────────────────────────────────────────────────
def _compute_beliefs(
    sm: StateManager,
    agent_intent: Dict[str, str],
) -> Tuple[Dict[str, Any], float]:
    """
    Light Theory of Mind: each agent infers what the other is targeting
    by observing their position and comparing to item locations.

    Returns:
        beliefs  : dict with cross-agent target predictions
        tom_reward: +0.05 if prediction matches actual intent, -0.02 if wrong
    """
    agent_ids   = list(sm.robots.keys())
    beliefs: Dict[str, Any] = {}
    tom_reward  = 0.0

    if len(agent_ids) < 2:
        return beliefs, tom_reward

    all_held = {item for r in sm.robots.values() for item in r["carrying"]}

    for observer in agent_ids:
        for target_agent in agent_ids:
            if observer == target_agent:
                continue

            target_robot = sm.robots[target_agent]
            tx, ty       = target_robot["pos"]

            # Predict: nearest unclaimed item from target agent's position
            best_item = None
            best_dist = float("inf")
            for item, loc in sm.inventory.items():
                if item in all_held:
                    continue
                dist = abs(loc[0] - tx) + abs(loc[1] - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_item = item

            # Ground truth: actual inferred intent of target_agent
            actual_intent = agent_intent.get(target_agent, "exploring")

            # Prediction label
            if target_robot["carrying"]:
                predicted_intent = "delivering"
            elif best_item is not None and best_dist <= 3:
                predicted_intent = "fetching_item"
            elif target_robot["battery"] <= 20:
                predicted_intent = "seeking_charge"
            else:
                predicted_intent = "exploring"

            key = f"{observer}_thinks_{target_agent}_target"
            beliefs[key] = {
                "predicted": predicted_intent,
                "actual":    actual_intent,
                "nearest_item_seen": best_item,
                "correct":   predicted_intent == actual_intent,
            }

            # Theory-of-mind reward signal
            if predicted_intent == actual_intent:
                tom_reward += 0.05
            else:
                tom_reward -= 0.02

    return beliefs, tom_reward


# ─────────────────────────────────────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────────────────────────────────────
class WarehouseEnv:
    """
    Multi-agent warehouse RL environment — v3 (Fleet AI edition).

    - Accepts ONE action per step, routed by action.agent_id
    - Fleet AI silently monitors + overrides before execution
    - Returns (Observation, Reward, done, info)
    - info["oversight"] exposes all Fleet AI metrics
    - info["beliefs"]   exposes Theory of Mind predictions
    - Backward-compatible: agent_id defaults to "agent1"

    Upgrade 1: True partial observability — each agent receives only
               items/obstacles within radius=2 of their own position.
    Upgrade 2: Theory of Mind — agents infer other agents' intent;
               correct predictions earn +0.05, wrong -0.02 per step.
    """

    def __init__(self) -> None:
        self.state_manager   = StateManager()
        self.fleet_ai        = FleetAI()
        self.done            = False
        self._current_task   = "easy"

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self, task: str = "easy", difficulty: int = 0) -> Observation:
        """Reset environment. difficulty=0 — static config; difficulty>=1 — dynamic (Theme 4)."""
        self._current_task = task
        self.done          = False
        self.fleet_ai.reset()
        self.state_manager.reset(task=task, difficulty=difficulty)
        # Return partial obs from agent1's perspective on reset
        return self._make_obs("agent1")

    # ── State ──────────────────────────────────────────────────────────────────
    def state(self) -> dict:
        return self.state_manager.get_state()

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action: Action):
        sm = self.state_manager
        sm.steps += 1
        sm.update_obstacles()

        agent_id          = action.agent_id
        reward_components: Dict[str, float] = {}
        raw_reward        = 0.0

        # ── Theme #1: Intent conflict detection BEFORE action ─────────────
        has_conflict, conflict_item, efficiency_split = _detect_intent_conflict(sm)
        if has_conflict:
            # Both agents chasing same item — soft penalty to discourage
            raw_reward += -0.05
            reward_components["competition_penalty"] = -0.05
        elif efficiency_split:
            # Agents split tasks efficiently — small bonus
            raw_reward += 0.1
            reward_components["task_split_bonus"] = 0.1

        # ── Feature 2: Infer intent for ALL agents before action fires ─────
        agent_intent: Dict[str, str] = {
            aid: _infer_intent(aid, sm) for aid in sm.robots
        }

        # ── Upgrade 2: Theory of Mind belief update + reward ──────────────
        beliefs, tom_reward = _compute_beliefs(sm, agent_intent)
        if tom_reward != 0.0:
            raw_reward += tom_reward
            reward_components["theory_of_mind"] = tom_reward

        # ── Phase 4: Fleet AI Intervention ────────────────────────────────
        final_action, was_overridden, explanation = self.fleet_ai.intervene(
            agent_id, action, sm
        )
        if was_overridden:
            print(f"[FLEET AI] {explanation}", flush=True)

        # ── Action Dispatch ────────────────────────────────────────────────
        if final_action.action_type == "move":
            success, reason = sm.move(agent_id, final_action.direction or "right")
            if success:
                raw_reward += -0.01
                reward_components["movement"] = -0.01
            elif reason == "agent_collision":
                raw_reward += -0.5
                reward_components["agent_collision"] = -0.5
            else:
                raw_reward += -0.3
                reward_components["collision"] = -0.3

        elif final_action.action_type == "pick":
            success, result = sm.pick(agent_id)
            if success:
                raw_reward += 0.3
                reward_components["pickup"] = 0.3
                # Phase 5: subtask reward
                raw_reward += 0.02
                reward_components["subtask"] = reward_components.get("subtask", 0.0) + 0.02

        elif final_action.action_type == "drop":
            completed, _ = sm.drop(agent_id)
            if completed > 0:
                raw_reward += completed * 0.5
                reward_components["delivery"] = completed * 0.5
                # Phase 1: correct sequencing bonus
                raw_reward += 0.3
                reward_components["sequencing"] = 0.3
                # Phase 5: subtask reward per completed order
                raw_reward += 0.02 * completed
                reward_components["subtask"] = reward_components.get("subtask", 0.0) + 0.02 * completed
            else:
                # Phase 1: penalty for dependency violation attempt
                robot_carrying = sm.robots.get(agent_id, {}).get("carrying", [])
                for order in sm.orders:
                    oid = order["id"]
                    if oid in sm.completed_orders:
                        continue
                    dep = order.get("depends_on")
                    if dep and dep not in sm.completed_orders:
                        if any(item in robot_carrying for item in order["items"]):
                            raw_reward += -0.2
                            reward_components["dependency_violation"] = -0.2
                            break

            # Full completion bonus
            if len(sm.completed_orders) == len(sm.orders):
                raw_reward += 1.0
                reward_components["completion_bonus"] = 1.0
                self.done = True

                # Cooperation bonus — both agents must have contributed
                if sm.check_cooperation_bonus():
                    raw_reward += 1.5
                    reward_components["cooperation_bonus"] = 1.5
                    sm.coordination_events += 1
                    collab_orders = list(sm.order_contributors.keys())
                    print(
                        f"[COOPERATION] Agents collaborated on order(s): {collab_orders}",
                        flush=True,
                    )

        elif final_action.action_type == "charge":
            success, reason = sm.charge(agent_id)
            if reason == "station_occupied":
                # Phase 2: charge conflict penalty
                raw_reward += -0.1
                reward_components["charge_conflict"] = -0.1
                print(
                    f"[CHARGE CONFLICT] {agent_id} failed — station busy with {sm.charging_agent}",
                    flush=True,
                )
            elif success:
                reward_components["charge"] = 0.0  # survival action — no reward/penalty

        # ── Phase 4: Oversight reward ──────────────────────────────────────
        if was_overridden and self.fleet_ai.efficiency_improvements > 0:
            raw_reward += 0.1
            reward_components["oversight"] = reward_components.get("oversight", 0.0) + 0.1

        # ── Theme #2: Planning failure penalty (drop failed due to dep) ────
        if final_action.action_type == "drop":
            robot_now = sm.robots.get(agent_id, {})
            if robot_now.get("carrying"):  # still carrying after drop attempt = dep blocked
                raw_reward += -0.05
                reward_components["planning_failure"] = reward_components.get("planning_failure", 0.0) - 0.05
                sm.planning_failures = getattr(sm, "planning_failures", 0) + 1

        # ── Feature 1: Prediction reward (+0.03 when prediction prevented failure)
        if self.fleet_ai._prediction_used and was_overridden:
            raw_reward += 0.03
            reward_components["prediction"] = reward_components.get("prediction", 0.0) + 0.03

        # ── Terminal: battery dead ─────────────────────────────────────────
        robot = sm.robots.get(agent_id, {})
        if robot.get("battery", 100) <= 0:
            raw_reward += -2.0
            reward_components["battery_dead"] = -2.0
            self.done = True

        # ── Terminal: step limit (Phase 5: extended) ───────────────────────
        if sm.steps >= _STEP_LIMITS.get(self._current_task, 150):
            self.done = True

        # ── Phase 2: coordination bonus (agents in different cells) ────────
        positions = [tuple(r["pos"]) for r in sm.robots.values()]
        if len(set(positions)) == len(positions):
            raw_reward += 0.05
            reward_components["coordination"] = reward_components.get("coordination", 0.0) + 0.05

        # ── Normalize ─────────────────────────────────────────────────────
        normalized = _normalize(raw_reward)

        # ── Full reward breakdown (19 components incl. Theory of Mind) ──────
        reward_breakdown: Dict[str, float] = {
            "movement":             reward_components.get("movement",             0.0),
            "collision":            reward_components.get("collision",            0.0),
            "agent_collision":      reward_components.get("agent_collision",      0.0),
            "pickup":               reward_components.get("pickup",               0.0),
            "delivery":             reward_components.get("delivery",             0.0),
            "sequencing":           reward_components.get("sequencing",           0.0),
            "dependency_violation": reward_components.get("dependency_violation", 0.0),
            "coordination":         reward_components.get("coordination",         0.0),
            "cooperation_bonus":    reward_components.get("cooperation_bonus",    0.0),
            "competition_penalty":  reward_components.get("competition_penalty",  0.0),
            "task_split_bonus":     reward_components.get("task_split_bonus",     0.0),
            "completion_bonus":     reward_components.get("completion_bonus",     0.0),
            "subtask":              reward_components.get("subtask",              0.0),
            "oversight":            reward_components.get("oversight",            0.0),
            "prediction":           reward_components.get("prediction",           0.0),
            "charge_conflict":      reward_components.get("charge_conflict",      0.0),
            "battery_dead":         reward_components.get("battery_dead",         0.0),
            "planning_failure":     reward_components.get("planning_failure",     0.0),
            # Upgrade 2: Theory of Mind
            "theory_of_mind":       reward_components.get("theory_of_mind",       0.0),
        }

        # ── Phase 4 + Feature 1&2: Oversight metrics (with intent) ────────
        oversight = self.fleet_ai.compute_oversight_metrics(sm, intent_analysis=agent_intent)

        # ── Info dict ─────────────────────────────────────────────────────
        remaining           = [o for o in sm.orders if o["id"] not in sm.completed_orders]
        cooperation_fired   = "cooperation_bonus" in reward_components
        n_orders            = max(len(sm.orders), 1)
        # Theme #2: long-horizon score = completed fraction weighted by depth
        long_horizon_score  = round(len(sm.completed_orders) / n_orders, 4)
        # Theme #3: belief_error — proxy: collisions while carrying (acts on stale info)
        robot_now           = sm.robots.get(agent_id, {})
        carrying_now        = robot_now.get("carrying", [])
        belief_error        = (sm.collisions + sm.agent_collisions) > 0 and bool(carrying_now)
        # Theme #4: curriculum_level = difficulty attr (0 when static)
        curriculum_level    = getattr(sm, "difficulty", 0)
        # Theme #1: coordination efficiency
        coordination_eff    = _compute_coordination_efficiency(sm)

        info: Dict[str, Any] = {
            "steps":                       sm.steps,
            "collisions":                  sm.collisions,
            "agent_collisions":            sm.agent_collisions,
            "completed_orders":            len(sm.completed_orders),
            "remaining_orders":            len(remaining),
            "success":                     self.done and len(sm.completed_orders) == len(sm.orders),
            "cooperation_bonus_triggered": cooperation_fired,
            "cooperation":                 cooperation_fired,
            "raw_reward":                  round(raw_reward, 6),
            "fleet_ai_intervened":         was_overridden,
            "fleet_ai_explanation":        self.fleet_ai._last_explanation or None,
            "subtasks_completed":          sm.subtasks_completed,
            # Theme #1
            "coordination_efficiency":     coordination_eff,
            "intent_conflict":             has_conflict,
            "conflict_item":               conflict_item,
            # Theme #2
            "long_horizon_score":          long_horizon_score,
            "planning_failures":           getattr(sm, "planning_failures", 0),
            # Theme #3
            "belief_error":                belief_error,
            # Theme #4
            "curriculum_level":            curriculum_level,
            # Agent intent
            "agent_intent":                agent_intent,
            # Upgrade 2: Theory of Mind beliefs
            "beliefs":                     beliefs,
            # Full reward breakdown
            "reward_breakdown":            reward_breakdown,
            # Snapshot for demo readability
            "carrying": {
                aid: list(r["carrying"]) for aid, r in sm.robots.items()
            },
            # Fleet AI dashboard
            "oversight": oversight,
            # Expert feedback
            "expert_feedback": _expert_feedback(agent_id, sm),
        }

        # Upgrade 1: True partial observability — last-acting agent's view
        self._validate_state()
        obs = self._make_obs(agent_id)
        return obs, Reward(value=normalized, breakdown=reward_components), self.done, info

    def _validate_state(self):
        robots = self.state_manager.robots

        # Check duplicate positions
        positions = [tuple(r["pos"]) for r in robots.values()]
        if len(positions) != len(set(positions)):
            print("⚠ WARNING: Duplicate robot positions detected")

        # Battery validation
        for agent_id, r in robots.items():
            if r["battery"] < 0:
                print(f"⚠ WARNING: {agent_id} battery below 0 → correcting")
                r["battery"] = 0

        # Carrying validation
        for agent_id, r in robots.items():
            valid_items = []
            for item in r.get("carrying", []):
                if item in ["item1", "item2", "item3"]:
                    valid_items.append(item)
                else:
                    print(f"⚠ WARNING: Invalid carrying state for {agent_id}")
            r["carrying"] = valid_items

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _make_obs(self, agent_id: str) -> Observation:
        """
        Upgrade 1 — True Partial Observability.
        Returns agent-specific Observation:
          - robots : ALL agents' positions/battery/carrying (global — allowed)
          - inventory : ONLY items within radius=2 of agent_id
          - obstacles : ONLY obstacles within radius=2 of agent_id
          - orders / goal / charge_station : global (agents must know task)
          - steps / completed_orders : global (shared knowledge)
        """
        sm    = self.state_manager
        pobs  = sm.get_partial_obs(agent_id)   # radius=2 local view
        full  = sm.get_state()                 # for shared fields

        return Observation(
            robots={
                aid: RobotState(
                    pos=rs["pos"],
                    battery=rs["battery"],
                    carrying=rs["carrying"],
                )
                for aid, rs in full["robots"].items()
            },
            # Upgrade 1: agent-specific partial inventory and obstacles
            inventory=pobs["visible_items"],
            obstacles=pobs["visible_obstacles"],
            orders=[Order(**o) for o in full["orders"]],
            goal=full["goal"],
            steps=full["steps"],
            completed_orders=full["completed_orders"],
            charge_station=full["charge_station"],
        )