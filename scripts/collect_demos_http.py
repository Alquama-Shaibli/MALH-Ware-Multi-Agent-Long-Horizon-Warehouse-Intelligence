#!/usr/bin/env python3
"""
scripts/collect_demos_http.py
==============================
Collect expert demonstration trajectories via the OpenEnv HTTP server.

Outputs JSONL to results/demos.jsonl (one JSON object per step).
Uses a heuristic policy — does NOT import warehouse_env.* directly.

Usage:
    # With server running on localhost:8000
    python scripts/collect_demos_http.py --episodes 50 --task hard

    # Judge mode with negotiation + program
    python scripts/collect_demos_http.py --mode negotiation --program-id hard_full \\
        --episodes 30 --task hard --base-url http://localhost:8000
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from client.openenv_client import OpenEnvClient


AGENTS = ["agent1", "agent2"]
MAX_STEPS = {"easy": 150, "medium": 200, "hard": 300, "judge": 300}


def _heuristic(agent_id: str, obs: dict, info: dict) -> dict:
    """
    Simple heuristic policy — navigate to nearest unclaimed item, then to goal.
    Returns a step kwargs dict for OpenEnvClient.step().
    """
    robots   = obs.get("robots", {})
    robot    = robots.get(agent_id, {})
    pos      = robot.get("pos", [0, 0])
    carrying = robot.get("carrying", [])
    battery  = robot.get("battery", 100)

    # Claim items in negotiation mode
    active_offers = info.get("active_offers", [])
    claim_map     = info.get("claim_map", {})

    # State from obs
    goal    = obs.get("goal", [5, 5])
    station = obs.get("charge_station", [0, 5])
    orders  = obs.get("orders", [])
    completed = obs.get("completed_orders", [])
    inventory = obs.get("inventory", {})

    def nav_dir(fx, fy):
        dx, dy = fx - pos[0], fy - pos[1]
        if abs(dx) >= abs(dy):
            return "down" if dx > 0 else "up"
        return "right" if dy > 0 else "left"

    # Charge if low
    if battery < 15 and pos != station:
        return {"agent_id": agent_id, "action_type": "move", "direction": nav_dir(*station)}
    if pos == station and battery < 50:
        return {"agent_id": agent_id, "action_type": "charge"}

    # Drop if at goal with items
    if pos == goal and carrying:
        return {"agent_id": agent_id, "action_type": "drop"}

    # Pick if standing on item
    for item, loc in inventory.items():
        if loc == pos and item not in [i for r in robots.values() for i in r.get("carrying", [])]:
            return {"agent_id": agent_id, "action_type": "pick"}

    # Navigate toward uncompleted order items
    pending_items = set()
    for o in orders:
        if o["id"] in completed:
            continue
        dep = o.get("depends_on")
        if dep and dep not in completed:
            continue
        for it in o["items"]:
            pending_items.add(it)

    all_carried = [i for r in robots.values() for i in r.get("carrying", [])]

    if carrying:
        return {"agent_id": agent_id, "action_type": "move", "direction": nav_dir(*goal)}

    targets = [(it, loc) for it, loc in inventory.items() if it not in all_carried and it in pending_items]
    if targets:
        # agent1 takes first, agent2 takes last
        tgt = targets[0][1] if agent_id == "agent1" else targets[-1][1]
        return {"agent_id": agent_id, "action_type": "move", "direction": nav_dir(*tgt)}

    return {"agent_id": agent_id, "action_type": "move", "direction": "right"}


def collect(
    base_url: str,
    task: str,
    mode: str,
    program_id: str | None,
    n_episodes: int,
    seed_start: int,
    out_path: str,
) -> int:
    client = OpenEnvClient(base_url)

    print(f"Waiting for server at {base_url}...")
    if not client.wait_for_server(max_wait=30):
        print("ERROR: Server not responding.", file=sys.stderr)
        sys.exit(1)
    print("Server ready.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_steps = 0
    max_s = MAX_STEPS.get(task, 300)

    with open(out_path, "w") as f:
        for ep in range(n_episodes):
            seed = seed_start + ep
            obs  = client.reset(task=task, seed=seed, mode=mode, program_id=program_id)
            done = False
            step = 0
            info: dict = {}

            while not done and step < max_s:
                step += 1
                for agent_id in AGENTS:
                    if done:
                        break
                    act_kwargs = _heuristic(agent_id, obs, info)
                    result = client.step(**act_kwargs)

                    record = {
                        "episode":    ep,
                        "seed":       seed,
                        "step":       step,
                        "agent_id":   agent_id,
                        "action":     act_kwargs,
                        "reward":     result.reward,
                        "done":       result.done,
                        # Observation summary (not full obs to keep file size manageable)
                        "robots":     result.observation.get("robots", {}),
                        "completed":  result.observation.get("completed_orders", []),
                        # Key info fields for SFT
                        "program_progress":   result.info.get("program_progress"),
                        "negotiation_events": result.info.get("negotiation_events", []),
                        "fleet_ai":           result.fleet_ai_intervention,
                        "fleet_ai_type":      result.fleet_ai_intervention_type,
                    }
                    f.write(json.dumps(record) + "\n")
                    total_steps += 1
                    obs  = result.observation
                    done = result.done
                    info = result.info

            print(f"  EP {ep+1:3d}/{n_episodes}  seed={seed}  steps={step}  "
                  f"done={done}  completed={info.get('completed_orders', 0)}")

    print(f"\nDone. {total_steps} steps written to {out_path}")
    return total_steps


def main():
    ap = argparse.ArgumentParser(description="Collect expert demos via HTTP")
    ap.add_argument("--base-url",   default="http://localhost:8000")
    ap.add_argument("--task",       default="hard", choices=["easy", "medium", "hard", "judge"])
    ap.add_argument("--mode",       default="default", choices=["default", "negotiation"])
    ap.add_argument("--program-id", default=None)
    ap.add_argument("--episodes",   type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out",        default="results/demos.jsonl")
    args = ap.parse_args()

    collect(
        base_url=args.base_url,
        task=args.task,
        mode=args.mode,
        program_id=args.program_id,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
