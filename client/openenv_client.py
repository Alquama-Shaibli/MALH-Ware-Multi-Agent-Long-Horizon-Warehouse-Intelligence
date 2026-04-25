"""
client/openenv_client.py
========================
HTTP client for the Smart Warehouse OpenEnv server.

Zero imports from warehouse_env.* — this is the judge-safe path.
Uses only `requests` (stdlib-adjacent) and `dataclasses`.

Usage:
    from client.openenv_client import OpenEnvClient

    client = OpenEnvClient("http://localhost:8000")
    obs   = client.reset(task="hard", program_id="hard_full", mode="negotiation")
    result = client.step("agent1", "move", direction="right")
    print(result.reward, result.done, result.info["program_progress"])
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── Response types ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    reward_breakdown: Dict[str, float]
    done: bool
    info: Dict[str, Any]

    @property
    def program_progress(self) -> Optional[Dict[str, Any]]:
        return self.info.get("program_progress")

    @property
    def negotiation_events(self) -> list:
        return self.info.get("negotiation_events", [])

    @property
    def fleet_ai_intervention(self) -> bool:
        return bool(self.info.get("fleet_ai_intervention"))

    @property
    def fleet_ai_intervention_type(self) -> str:
        return self.info.get("fleet_ai_intervention_type", "")


# ── Client ─────────────────────────────────────────────────────────────────────

class OpenEnvClient:
    """
    Typed HTTP client for the Smart Warehouse OpenEnv server.

    All methods raise RuntimeError on HTTP error or if requests is not installed.
    Retries once on connection error (server may be starting up).
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        if not _HAS_REQUESTS:
            raise RuntimeError("requests is not installed. Run: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    # ── Health ─────────────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        return self._get("/")

    def wait_for_server(self, max_wait: int = 30) -> bool:
        """Poll / until server responds. Returns True if ready."""
        start = time.time()
        while time.time() - start < max_wait:
            try:
                self.health()
                return True
            except Exception:
                time.sleep(1.0)
        return False

    # ── Core endpoints ─────────────────────────────────────────────────────────
    def reset(
        self,
        task: str = "easy",
        seed: Optional[int] = None,
        mode: str = "default",
        program_id: Optional[str] = None,
        difficulty: int = 0,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"task": task, "mode": mode, "difficulty": difficulty}
        if seed is not None:
            body["seed"] = seed
        if program_id:
            body["program_id"] = program_id
        return self._post("/reset", body)

    def step(
        self,
        agent_id: str = "agent1",
        action_type: str = "move",
        direction: Optional[str] = None,
        item_id: Optional[str] = None,
        target_agent: Optional[str] = None,
        offer_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        body: Dict[str, Any] = {"agent_id": agent_id, "action_type": action_type}
        if direction:    body["direction"]    = direction
        if item_id:      body["item_id"]      = item_id
        if target_agent: body["target_agent"] = target_agent
        if offer_id:     body["offer_id"]     = offer_id
        if payload:      body["payload"]      = payload
        raw = self._post("/step", body)
        return StepResult(
            observation=raw.get("observation", {}),
            reward=float(raw.get("reward", 0.0)),
            reward_breakdown=raw.get("reward_breakdown", {}),
            done=bool(raw.get("done", False)),
            info=raw.get("info", {}),
        )

    def state(self) -> Dict[str, Any]:
        return self._get("/state")

    def oversight(self) -> Dict[str, Any]:
        return self._get("/oversight")

    def program(self) -> Dict[str, Any]:
        return self._get("/program")

    # ── Convenience helpers ────────────────────────────────────────────────────
    def run_episode(
        self,
        task: str = "easy",
        seed: Optional[int] = None,
        mode: str = "default",
        program_id: Optional[str] = None,
        difficulty: int = 0,
        policy_fn=None,
        max_steps: int = 300,
    ) -> Dict[str, Any]:
        """
        Run one full episode using `policy_fn(obs, info) -> (agent_id, action_type, kwargs)`.
        If policy_fn is None, uses a simple move-right no-op.
        Returns summary dict: total_reward, steps, completed_orders, interventions, etc.
        """
        self.reset(task=task, seed=seed, mode=mode, program_id=program_id, difficulty=difficulty)
        total_reward  = 0.0
        steps         = 0
        interventions = 0
        done          = False
        info: Dict[str, Any] = {}

        agents = ["agent1", "agent2"]

        while not done and steps < max_steps:
            steps += 1
            for aid in agents:
                if done:
                    break
                if policy_fn:
                    act = policy_fn(aid, info)
                    result = self.step(**act) if isinstance(act, dict) else self.step(aid, *act)
                else:
                    result = self.step(aid, "move", direction="right")

                total_reward += result.reward
                done          = result.done
                info          = result.info
                if result.fleet_ai_intervention:
                    interventions += 1

        return {
            "total_reward":      total_reward,
            "steps":             steps,
            "completed_orders":  info.get("completed_orders", 0),
            "interventions":     interventions,
            "program_progress":  info.get("program_progress"),
            "negotiation_events": len(info.get("negotiation_events", [])),
            "done":              done,
        }

    # ── Internal HTTP helpers ──────────────────────────────────────────────────
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{self.base_url}{path}", json=body, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                raise RuntimeError(f"Cannot connect to server at {self.base_url}")
        return {}

    def _get(self, path: str) -> Dict[str, Any]:
        for attempt in range(2):
            try:
                resp = requests.get(
                    f"{self.base_url}{path}", timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                raise RuntimeError(f"Cannot connect to server at {self.base_url}")
        return {}
