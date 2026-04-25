import random
from typing import Any, Dict, List, Optional, Tuple


class StateManager:
    """
    Multi-agent warehouse state manager — v3 (Fleet AI edition).

    New in v3:
    - Phase 2: Charge station contention (only 1 agent at a time)
    - Phase 3: Partial observability per agent (radius=2)
    - Phase 4: Oversight metrics (idle_steps, coordination_events, charge_conflicts)
    - Phase 5: Subtask tracking for long-horizon reward shaping
    """

    # ── Task Configurations ────────────────────────────────────────────────────
    _CONFIGS: Dict[str, Dict[str, Any]] = {
        "easy": {
            "grid":        6,
            "battery":     100,
            "n_obstacles": 0,
            "items":       ["item1", "item2"],
            "orders": [
                {"id": "order1", "items": ["item1"], "priority": 1, "depends_on": None},
            ],
        },
        "medium": {
            "grid":        8,
            "battery":     80,
            "n_obstacles": 2,
            "items":       ["item1", "item2", "item3"],
            "orders": [
                {"id": "order1", "items": ["item1", "item2"], "priority": 1, "depends_on": None},
                {"id": "order2", "items": ["item3"],          "priority": 2, "depends_on": None},
            ],
        },
        "hard": {
            "grid":        10,
            "battery":     50,
            "n_obstacles": 4,
            "items":       ["item1", "item2", "item3", "item4"],
            "orders": [
                {"id": "order1", "items": ["item1", "item2"], "priority": 1, "depends_on": None},
                {"id": "order2", "items": ["item3"],          "priority": 2, "depends_on": "order1"},
                {"id": "order3", "items": ["item4"],          "priority": 3, "depends_on": "order2"},
            ],
        },
    }

    _RAW_ITEM_POS: List[Tuple[float, float]] = [
        (0.33, 0.33),
        (0.50, 0.17),
        (0.67, 0.67),
        (0.17, 0.67),
        (0.50, 0.50),  # 5th position (centre)
    ]

    def __init__(self) -> None:
        self.grid_size: int = 6
        self.task: str = "easy"
        self.difficulty: int = 0        # Theme 4: adaptive difficulty level
        self.robots: Dict[str, Dict[str, Any]] = {}
        self.inventory: Dict[str, List[int]] = {}
        self.orders: List[Dict[str, Any]] = []
        self.completed_orders: List[str] = []
        self.order_contributors: Dict[str, set] = {}
        self.goal: List[int] = [5, 5]
        self.charge_station: List[int] = [0, 5]
        self.obstacles: List[Tuple[int, int]] = []
        self.steps: int = 0
        self.collisions: int = 0
        self.agent_collisions: int = 0
        # Phase 2 — charge contention
        self.charging_agent: Optional[str] = None
        self.charge_conflicts: int = 0
        # Phase 4 — oversight metrics
        self.idle_steps: Dict[str, int] = {}
        self.coordination_events: int = 0
        self.subtasks_completed: int = 0
        self.intervention_log: List[str] = []
        # Theme #2/#4: long-horizon + self-improvement tracking
        self.planning_failures: int = 0
        self.success_streak: int = 0

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self, task: str = "easy", difficulty: int = 0) -> Dict[str, Any]:
        """
        Reset environment.
        difficulty=0 (default) — uses static config (backward-compatible).
        difficulty>=1           — uses dynamic task generation (Theme 4).
        """
        self.task       = task
        self.difficulty = difficulty

        if difficulty > 0:
            self._dynamic_reset(task, difficulty)
        else:
            self._static_reset(task)

        # Common reset of counters
        self.steps            = 0
        self.collisions       = 0
        self.agent_collisions = 0
        self.charging_agent   = None
        self.charge_conflicts = 0
        self.idle_steps          = {aid: 0 for aid in self.robots}
        self.coordination_events = 0
        self.subtasks_completed  = 0
        self.intervention_log    = []
        self.planning_failures   = 0
        # Note: success_streak intentionally NOT reset here — persists across episodes

        return self.get_state()

    def _static_reset(self, task: str) -> None:
        """Config-driven reset. Easy mode randomises item count (2-5) each episode."""
        cfg = self._CONFIGS.get(task, self._CONFIGS["easy"])
        self.grid_size = cfg["grid"]
        g = self.grid_size

        self.robots = {
            "agent1": {"pos": [0,     0], "battery": float(cfg["battery"]), "carrying": []},
            "agent2": {"pos": [g - 1, 0], "battery": float(cfg["battery"]), "carrying": []},
        }
        self.goal           = [g - 1, g - 1]
        self.charge_station = [0,     g - 1]

        # ── Random item count for easy mode (2-5), fixed for others ──────
        if task == "easy":
            n_items = random.randint(2, 5)
            item_names = [f"item{i+1}" for i in range(n_items)]
        else:
            item_names = cfg["items"]
            n_items = len(item_names)

        # Place items at unique grid positions (avoid corners/protected cells)
        protected = {(0, 0), (g-1, 0), (g-1, g-1), (0, g-1)}
        self.inventory = {}
        for i, item_name in enumerate(item_names):
            # Try preset position first, then fall back to random
            rx, ry = self._RAW_ITEM_POS[i % len(self._RAW_ITEM_POS)]
            x = max(1, min(g - 2, int(rx * g)))
            y = max(1, min(g - 2, int(ry * g)))
            # If collision with existing item, find random free cell
            attempts = 0
            while ([x, y] in self.inventory.values() or (x, y) in protected) and attempts < 50:
                x = random.randint(1, g - 2)
                y = random.randint(1, g - 2)
                attempts += 1
            self.inventory[item_name] = [x, y]

        # ── Auto-generate one order per item (all independent, no deps) ──
        if task == "easy":
            self.orders = [
                {"id": f"order{i+1}", "items": [name], "priority": i+1, "depends_on": None}
                for i, name in enumerate(item_names)
            ]
        else:
            self.orders = [dict(o) for o in cfg["orders"]]

        self.completed_orders   = []
        self.order_contributors = {}
        self._place_obstacles(n=cfg["n_obstacles"])

    def _dynamic_reset(self, task: str, difficulty: int) -> None:
        """Theme 4: dynamic task generation driven by difficulty level."""
        # Base grid from task, enlarged by difficulty
        base_cfg   = self._CONFIGS.get(task, self._CONFIGS["easy"])
        base_grid  = base_cfg["grid"]
        g          = min(base_grid + difficulty // 2, 14)   # cap at 14
        self.grid_size = g

        # Battery decreases slightly with difficulty (harder = less battery)
        battery = max(30.0, float(base_cfg["battery"]) - difficulty * 3)

        self.robots = {
            "agent1": {"pos": [0,     0], "battery": battery, "carrying": []},
            "agent2": {"pos": [g - 1, 0], "battery": battery, "carrying": []},
        }
        self.goal           = [g - 1, g - 1]
        self.charge_station = [0,     g - 1]

        # Dynamic items: 3 + difficulty, capped at 8
        n_items = min(3 + difficulty, 8)
        self.inventory = {}
        protected_positions = {(0, 0), (g - 1, 0), (g - 1, g - 1), (0, g - 1)}
        for i in range(n_items):
            item_name = f"item{i + 1}"
            for _ in range(200):
                x = random.randint(1, g - 2)
                y = random.randint(1, g - 2)
                if (x, y) not in protected_positions and [x, y] not in self.inventory.values():
                    self.inventory[item_name] = [x, y]
                    protected_positions.add((x, y))
                    break

        # Dynamic orders: 2 + difficulty // 2, capped at n_items
        n_orders = min(2 + difficulty // 2, n_items)
        item_names = list(self.inventory.keys())
        random.shuffle(item_names)
        self.orders = []
        self.completed_orders   = []
        self.order_contributors = {}
        for oi in range(n_orders):
            oid     = f"order{oi + 1}"
            # Each order claims 1 item (simple, avoids multi-item deadlocks)
            item    = item_names[oi % len(item_names)]
            # depends_on: chain orders after difficulty >= 3
            dep     = f"order{oi}" if (oi > 0 and difficulty >= 3) else None
            self.orders.append({
                "id":         oid,
                "items":      [item],
                "priority":   oi + 1,
                "depends_on": dep,
            })

        # Dynamic obstacles: 2 + difficulty, capped at grid cells // 4
        n_obstacles = min(2 + difficulty, (g * g) // 4)
        self._place_obstacles(n=n_obstacles)


    # ── Obstacle Management ────────────────────────────────────────────────────
    def _protected_cells(self) -> set:
        protected = set()
        protected.add(tuple(self.goal))
        protected.add(tuple(self.charge_station))
        for r in self.robots.values():
            protected.add(tuple(r["pos"]))
        for pos in self.inventory.values():
            protected.add(tuple(pos))
        return protected

    def _place_obstacles(self, n: int) -> None:
        protected = self._protected_cells()
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < n and attempts < 200:
            pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if pos not in protected and pos not in self.obstacles:
                self.obstacles.append(pos)
            attempts += 1

    def update_obstacles(self) -> None:
        """Dynamic obstacles + stochastic item relocation (Theme #3 world model)."""
        if self.task == "hard" or self.difficulty >= 3:
            n = (
                self._CONFIGS["hard"]["n_obstacles"]
                if self.difficulty == 0
                else min(2 + self.difficulty, (self.grid_size * self.grid_size) // 4)
            )
            self._place_obstacles(n=n)
        # Theme #3: stochastic item relocation (1% probability per item per step)
        self._maybe_relocate_items()

    def _maybe_relocate_items(self, prob: float = 0.01) -> None:
        """With small probability, move an unclaimed inventory item to a new random cell."""
        protected = self._protected_cells()
        all_held  = {item for r in self.robots.values() for item in r["carrying"]}
        for item, loc in list(self.inventory.items()):
            if item in all_held:
                continue
            if random.random() < prob:
                for _ in range(50):
                    nx = random.randint(1, self.grid_size - 2)
                    ny = random.randint(1, self.grid_size - 2)
                    if (nx, ny) not in protected and [nx, ny] not in self.inventory.values():
                        self.inventory[item] = [nx, ny]
                        break

    # ── Movement ──────────────────────────────────────────────────────────────
    def move(self, agent_id: str, direction: str) -> Tuple[bool, str]:
        robot = self.robots.get(agent_id)
        if robot is None:
            return False, "invalid_agent"

        x, y = robot["pos"]
        deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        dx, dy = deltas.get(direction, (0, 0))
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            self.collisions += 1
            self.idle_steps[agent_id] = self.idle_steps.get(agent_id, 0) + 1
            return False, "wall"

        if (nx, ny) in self.obstacles:
            self.collisions += 1
            self.idle_steps[agent_id] = self.idle_steps.get(agent_id, 0) + 1
            return False, "obstacle"

        for aid, r in self.robots.items():
            if aid != agent_id and r["pos"] == [nx, ny]:
                self.agent_collisions += 1
                self.idle_steps[agent_id] = self.idle_steps.get(agent_id, 0) + 1
                return False, "agent_collision"


        robot["pos"] = [nx, ny]
        robot["battery"] = max(0.0, robot["battery"] - 1.0)
        # Release charge lock if moving away from station
        if self.charging_agent == agent_id:
            self.charging_agent = None
        return True, "ok"

    # ── Pick ──────────────────────────────────────────────────────────────────
    def pick(self, agent_id: str) -> Tuple[bool, str]:
        robot = self.robots.get(agent_id)
        if robot is None:
            return False, "invalid_agent"

        pos = robot["pos"]
        for item, loc in self.inventory.items():
            if loc != pos:
                continue
            for r in self.robots.values():
                if item in r["carrying"]:
                    return False, "already_held"
            robot["carrying"].append(item)
            self.subtasks_completed += 1
            return True, item

        self.idle_steps[agent_id] = self.idle_steps.get(agent_id, 0) + 1
        return False, "no_item"

    # ── Drop ──────────────────────────────────────────────────────────────────
    def drop(self, agent_id: str) -> Tuple[int, bool]:
        robot = self.robots.get(agent_id)
        if robot is None or robot["pos"] != self.goal:
            return 0, False

        completed = 0
        for order in self.orders:
            oid = order["id"]
            if oid in self.completed_orders:
                continue

            # Phase 1: Dependency gate
            dep = order.get("depends_on")
            if dep and dep not in self.completed_orders:
                continue

            if all(item in robot["carrying"] for item in order["items"]):
                self.completed_orders.append(oid)
                completed += 1
                self.subtasks_completed += 1

                if oid not in self.order_contributors:
                    self.order_contributors[oid] = set()
                self.order_contributors[oid].add(agent_id)

                for item in order["items"]:
                    if item in robot["carrying"]:
                        robot["carrying"].remove(item)
                    # Remove from inventory so it doesn't reappear as unclaimed
                    self.inventory.pop(item, None)

        # Theme #4: increment success streak when all orders done
        if len(self.completed_orders) == len(self.orders) and self.orders:
            self.success_streak += 1

        return completed, False

    # ── Charge (Phase 2: contention) ──────────────────────────────────────────
    def charge(self, agent_id: str) -> Tuple[bool, str]:
        """
        Only one agent may charge at a time.
        Returns (success, reason): reason in {ok, not_at_station, station_occupied}
        """
        robot = self.robots.get(agent_id)
        if not robot or robot["pos"] != self.charge_station:
            return False, "not_at_station"

        if self.charging_agent is not None and self.charging_agent != agent_id:
            self.charge_conflicts += 1
            return False, "station_occupied"

        self.charging_agent = agent_id
        robot["battery"] = 100.0
        return True, "ok"

    # ── Cooperation Check ─────────────────────────────────────────────────────
    def check_cooperation_bonus(self) -> bool:
        if not self.order_contributors:
            return False
        required_agents = set(self.robots.keys())
        all_contributors: set = set()
        for contributors in self.order_contributors.values():
            all_contributors |= contributors
        return required_agents.issubset(all_contributors)

    # ── Partial Observability (Phase 3) ───────────────────────────────────────
    def get_partial_obs(self, agent_id: str) -> Dict[str, Any]:
        """Vision radius = 2 cells around agent."""
        robot = self.robots.get(agent_id, {})
        rx, ry = robot.get("pos", [0, 0])

        visible_items = {
            item: loc
            for item, loc in self.inventory.items()
            if abs(loc[0] - rx) <= 2 and abs(loc[1] - ry) <= 2
        }
        visible_obstacles = [
            list(obs)
            for obs in self.obstacles
            if abs(obs[0] - rx) <= 2 and abs(obs[1] - ry) <= 2
        ]
        return {
            "robot":             robot,
            "visible_items":     visible_items,
            "visible_obstacles": visible_obstacles,
            "goal":              self.goal,
        }

    # ── Full State ────────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        return {
            "robots": {
                aid: {
                    "pos":      list(r["pos"]),
                    "battery":  r["battery"],
                    "carrying": list(r["carrying"]),
                }
                for aid, r in self.robots.items()
            },
            "inventory":        {k: list(v) for k, v in self.inventory.items()},
            "orders":           [dict(o) for o in self.orders],
            "goal":             list(self.goal),
            "obstacles":        [list(o) for o in self.obstacles],
            "steps":            self.steps,
            "completed_orders": list(self.completed_orders),
            "charge_station":   list(self.charge_station),
            "grid_size":        self.grid_size,
        }