import random
import matplotlib.pyplot as plt
import numpy as np

from warehouse_env.env_core import WarehouseEnv
from warehouse_env.models import Action

# -----------------------------
# Simple Learning Policy (Tabular Q-Learning)
# -----------------------------
class SimplePolicy:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.actions = [
            ("move", "up"),
            ("move", "down"),
            ("move", "left"),
            ("move", "right"),
            ("pick", None),
            ("drop", None),
            ("charge", None)
        ]
        self.q_values = {}
        self.epsilon = 1.0  # Start fully exploring
        self.epsilon_decay = 0.92
        self.min_epsilon = 0.05
        self.lr = 0.2
        self.gamma = 0.95

    def _get_state(self, obs):
        """Improved 5-tuple state: (dx, dy, carrying, obstacle_nearby, low_battery)"""
        if self.agent_id not in obs.robots:
            return (0, 0, False, False, False)
        robot = obs.robots[self.agent_id]
        is_carrying = len(robot.carrying) > 0
        rx, ry = robot.pos
        battery = robot.battery

        # Direction to nearest target
        if is_carrying:
            tx, ty = obs.goal
        else:
            items = list(obs.inventory) if isinstance(obs.inventory, list) else list(obs.inventory.values())
            if items:
                item = items[0]
                if isinstance(item, (list, tuple)): tx, ty = item[0], item[1]
                elif isinstance(item, dict) and 'pos' in item: tx, ty = item['pos']
                else: tx, ty = item.pos
            else:
                tx, ty = rx, ry

        dx, dy = tx - rx, ty - ry
        # Clamp to -1/0/+1 direction signal
        sdx = max(-1, min(1, dx))
        sdy = max(-1, min(1, dy))

        # Obstacle adjacency check
        obstacle_nearby = False
        if hasattr(obs, 'obstacles') and obs.obstacles:
            for obs_pos in obs.obstacles:
                ox, oy = obs_pos if isinstance(obs_pos, (list, tuple)) else (obs_pos['pos'] if isinstance(obs_pos, dict) and 'pos' in obs_pos else obs_pos.pos)
                if abs(ox - rx) <= 1 and abs(oy - ry) <= 1:
                    obstacle_nearby = True
                    break

        low_battery = battery < 20
        return (sdx, sdy, is_carrying, obstacle_nearby, low_battery)

    def _is_safe(self, action_tuple, obs, grid_size=10):
        """Filter out moves that walk into walls or obstacles."""
        if action_tuple[0] != "move":
            return True  # non-move actions are always safe to attempt
        if self.agent_id not in obs.robots:
            return True
        robot = obs.robots[self.agent_id]
        rx, ry = robot.pos
        direction = action_tuple[1]
        nx, ny = rx, ry
        if direction == "up":    nx -= 1
        elif direction == "down":  nx += 1
        elif direction == "left":  ny -= 1
        elif direction == "right": ny += 1
        # Wall check
        if not (0 <= nx < grid_size and 0 <= ny < grid_size):
            return False
        # Obstacle check
        if hasattr(obs, 'obstacles') and obs.obstacles:
            for obs_pos in obs.obstacles:
                ox, oy = obs_pos if isinstance(obs_pos, (list, tuple)) else (obs_pos['pos'] if isinstance(obs_pos, dict) and 'pos' in obs_pos else obs_pos.pos)
                if ox == nx and oy == ny:
                    return False
        return True

    def choose_action(self, obs):
        state = self._get_state(obs)
        if state not in self.q_values:
            self.q_values[state] = {a: 0.0 for a in self.actions}

        # Force pickup if agent is on an item
        if self.agent_id in obs.robots:
            robot = obs.robots[self.agent_id]
            rx, ry = robot.pos
            items = list(obs.inventory) if isinstance(obs.inventory, list) else list(obs.inventory.values())
            for item in items:
                if isinstance(item, (list, tuple)): ix, iy = item[0], item[1]
                elif isinstance(item, dict) and 'pos' in item: ix, iy = item['pos']
                else: ix, iy = item.pos
                if rx == ix and ry == iy and not len(robot.carrying) > 0:
                    return Action(agent_id=self.agent_id, action_type="pick", direction=None)

            # Force drop if agent is at goal and carrying
            gx, gy = obs.goal
            if len(robot.carrying) > 0 and rx == gx and ry == gy:
                return Action(agent_id=self.agent_id, action_type="drop", direction=None)

        # Safe action filter
        safe_actions = [a for a in self.actions if self._is_safe(a, obs)]
        if not safe_actions:
            safe_actions = self.actions

        if random.random() < self.epsilon:
            action_tuple = random.choice(safe_actions)
        else:
            action_tuple = max(safe_actions, key=lambda a: self.q_values[state].get(a, 0.0))

        return Action(
            agent_id=self.agent_id,
            action_type=action_tuple[0],
            direction=action_tuple[1]
        )

    def update(self, obs, action_obj, reward_val, next_obs):
        state = self._get_state(obs)
        next_state = self._get_state(next_obs)
        action_tuple = (action_obj.action_type, action_obj.direction)
        
        if state not in self.q_values:
            self.q_values[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_values:
            self.q_values[next_state] = {a: 0.0 for a in self.actions}
            
        best_next_q = max(self.q_values[next_state].values())
        current_q = self.q_values[state][action_tuple]
        
        # Q-learning update
        self.q_values[state][action_tuple] = current_q + self.lr * (reward_val - current_q + self.gamma * best_next_q)

    def end_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# -----------------------------
# Online RL Loop
# -----------------------------
def run_online_rl():
    env = WarehouseEnv()
    
    # We train Agent 1. Agent 2 does random walks to act as noise.
    agent1 = SimplePolicy("agent1")
    agent2 = SimplePolicy("agent2")

    rewards_per_episode = []
    collision_rates = []
    delivery_rates = []
    coordination_rates = []
    
    print("Starting Online RL Training (50 Episodes)...")

    for episode in range(50):
        # Force easy task to ensure it converges in 50 episodes
        obs = env.reset(task="easy")
        total_reward = 0
        ep_collisions = 0
        ep_deliveries = 0
        ep_coordination = 0

        for step in range(200):  # More steps per episode for better learning
            # Agent 1 Learning — uses Q-policy with safe action filter
            a1_action = agent1.choose_action(obs)

            # Agent 2 also uses policy (not pure random) to reduce wall-bouncing noise
            a2_action = agent2.choose_action(obs)

            # Execute Agent 1
            next_obs, reward_obj, done, info = env.step(a1_action)

            # ── Shaped reward (on top of env reward) ──────────────────────────
            r = reward_obj.value
            r -= 0.05  # small movement cost (was causing 0 gradient)
            breakdown = info.get("reward_breakdown", {})
            if breakdown.get("collision", 0) < 0:
                r -= 0.7   # strong collision penalty
                ep_collisions += 1
            if breakdown.get("pickup", 0) > 0:
                r += 1.5   # strong pickup reward
            if breakdown.get("delivery", 0) > 0:
                r += 4.0   # very strong delivery reward
                ep_deliveries += 1
            if "coordination_efficiency" in info:
                ep_coordination = info["coordination_efficiency"]

            agent1.update(obs, a1_action, r, next_obs)
            total_reward += r
            print(f"[Ep {episode+1} Step {step}] State: {agent1._get_state(obs)} | Action: ({a1_action.action_type},{a1_action.direction}) | Reward: {r:.3f}")

            obs = next_obs
            if done: break

            # Execute Agent 2
            next_obs, reward_obj, done, info = env.step(a2_action)
            r2 = reward_obj.value
            breakdown2 = info.get("reward_breakdown", {})
            if breakdown2.get("collision", 0) < 0: ep_collisions += 1
            if breakdown2.get("delivery", 0) > 0: ep_deliveries += 1
            agent2.update(obs, a2_action, r2, next_obs)

            obs = next_obs
            if done: break

        agent1.end_episode()
        agent2.end_episode()

        rewards_per_episode.append(total_reward)
        collision_rates.append(ep_collisions)
        delivery_rates.append(ep_deliveries)
        coordination_rates.append(ep_coordination)
        
        print(f"\n[Episode {episode+1:02d}]")
        print(f"Reward: {total_reward:.2f}")
        print(f"Epsilon: {agent1.epsilon:.2f}")
        print(f"Status: {'LEARNING' if agent1.epsilon > 0.3 else 'OPTIMIZING'}")

    print("\n=== TRAINING SUMMARY ===")
    print(f"Start Reward : {rewards_per_episode[0]:.2f}")
    print(f"Final Reward : {rewards_per_episode[-1]:.2f}")
    print(f"Improvement  : {rewards_per_episode[-1] - rewards_per_episode[0]:.2f}\n")

    # Save Q-table to disk for the UI to use
    import pickle
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent1.q_values, f)
    print("Saved Q-table to q_table.pkl")

    return rewards_per_episode, collision_rates, delivery_rates, coordination_rates

# -----------------------------
# Plot Results
# -----------------------------
def plot_results(metrics):
    rewards, collisions, deliveries, coordination = metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Reward Curve
    smoothed_rewards = [np.mean(rewards[max(0, i-5):i+1]) for i in range(len(rewards))]
    axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    axes[0, 0].plot(smoothed_rewards, color='blue', linewidth=2, marker='o', label='Smoothed Trend')
    axes[0, 0].set_title("Reward Curve", fontweight='bold')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Collision Rate
    axes[0, 1].plot(collisions, color='red', linewidth=2, marker='x')
    axes[0, 1].set_title("Collision Rate", fontweight='bold')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total Collisions")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Delivery Rate
    axes[1, 0].plot(deliveries, color='green', linewidth=2, marker='^')
    axes[1, 0].set_title("Delivery Rate", fontweight='bold')
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Items Delivered")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 4. Coordination Efficiency
    axes[1, 1].plot(coordination, color='purple', linewidth=2, marker='s')
    axes[1, 1].set_title("Coordination Efficiency", fontweight='bold')
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Efficiency Score")
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle("Online RL Learning Metrics (Tabular Q-Learning)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("online_rl_curve.png", dpi=150)
    print("Online RL completed - metrics grid saved as 'online_rl_curve.png'")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    metrics = run_online_rl()
    plot_results(metrics)
