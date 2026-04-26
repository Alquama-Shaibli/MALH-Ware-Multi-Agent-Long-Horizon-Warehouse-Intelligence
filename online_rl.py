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
        
        self.pos_history = []
        self.action_history = []

    def _get_state(self, obs):
        """Improved 8-tuple state: (agent_id, dx, dy, carrying, obstacle_nearby, low_battery, ndx, ndy)"""
        if self.agent_id not in obs.robots:
            return (self.agent_id, 0, 0, False, False, False, 0, 0)
        robot = obs.robots[self.agent_id]
        is_carrying = len(robot.carrying) > 0
        rx, ry = robot.pos
        battery = robot.battery

        # Direction to nearest target with task partition
        if is_carrying:
            tx, ty = obs.goal
        else:
            items = list(obs.inventory) if isinstance(obs.inventory, list) else list(obs.inventory.values())
            if items:
                # Agent 1 -> target closest (index 0). Agent 2 -> target second closest (index 1 if available)
                if self.agent_id == "agent2" and len(items) > 1:
                    item = items[1]
                else:
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
        
        # Nearby agent check
        ndx, ndy = 0, 0
        for other_id, other_robot in obs.robots.items():
            if other_id != self.agent_id:
                orx, ory = other_robot.pos
                odx, ody = orx - rx, ory - ry
                if abs(odx) <= 2 and abs(ody) <= 2:
                    ndx = max(-1, min(1, odx))
                    ndy = max(-1, min(1, ody))
                break

        return (self.agent_id, sdx, sdy, is_carrying, obstacle_nearby, low_battery, ndx, ndy)

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
        
        if hasattr(obs, 'grid_size'):
            grid_size = obs.grid_size
            
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

    def choose_action(self, obs, force_diverge=False):
        state = self._get_state(obs)
        if state not in self.q_values:
            self.q_values[state] = {a: 0.0 for a in self.actions}

        if self.agent_id in obs.robots:
            robot = obs.robots[self.agent_id]
            rx, ry = robot.pos
            
            # Position tracking for loop detection
            self.pos_history.append((rx, ry))
            if len(self.pos_history) > 4:
                self.pos_history.pop(0)

            is_looping = False
            if len(self.pos_history) == 4:
                if self.pos_history[0] == self.pos_history[2] and self.pos_history[1] == self.pos_history[3] and self.pos_history[0] != self.pos_history[1]:
                    is_looping = True
            
            # Force pickup if agent is on an item
            items = list(obs.inventory) if isinstance(obs.inventory, list) else list(obs.inventory.values())
            for item in items:
                if isinstance(item, (list, tuple)): ix, iy = item[0], item[1]
                elif isinstance(item, dict) and 'pos' in item: ix, iy = item['pos']
                else: ix, iy = item.pos
                if rx == ix and ry == iy and not len(robot.carrying) > 0:
                    action_str = "pick"
                    self.action_history.append(action_str)
                    if len(self.action_history) > 3: self.action_history.pop(0)
                    return Action(agent_id=self.agent_id, action_type="pick", direction=None)

            # Force drop if agent is at goal and carrying
            gx, gy = obs.goal
            if len(robot.carrying) > 0 and rx == gx and ry == gy:
                action_str = "drop"
                self.action_history.append(action_str)
                if len(self.action_history) > 3: self.action_history.pop(0)
                return Action(agent_id=self.agent_id, action_type="drop", direction=None)

        # Safe action filter
        safe_actions = [a for a in self.actions if self._is_safe(a, obs)]
        if not safe_actions:
            safe_actions = self.actions

        # Action diversity: prevent repeating identical action > 3 times
        repeating_action = False
        if len(self.action_history) == 3:
            if self.action_history[0] == self.action_history[1] == self.action_history[2]:
                repeating_action = True

        if random.random() < self.epsilon or force_diverge or is_looping or repeating_action:
            action_tuple = random.choice(safe_actions)
        else:
            action_tuple = max(safe_actions, key=lambda a: self.q_values[state].get(a, 0.0))

        action_str = f"{action_tuple[0]}_{action_tuple[1]}"
        self.action_history.append(action_str)
        if len(self.action_history) > 3:
            self.action_history.pop(0)
            
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
        self.pos_history = []
        self.action_history = []

# -----------------------------
# Online RL Loop
# -----------------------------
def run_online_rl():
    env = WarehouseEnv()
    
    agent1 = SimplePolicy("agent1")
    agent2 = SimplePolicy("agent2")

    rewards_per_episode = []
    collision_rates = []
    delivery_rates = []
    coordination_rates = []
    
    print("Starting Online RL Training (50 Episodes)...")

    for episode in range(50):
        obs = env.reset(task="easy")
        total_reward = 0
        ep_collisions = 0
        ep_deliveries = 0
        ep_coordination = 0
        
        a1_diverge = False
        a2_diverge = False

        for step in range(200):
            # Agent 1
            a1_action = agent1.choose_action(obs, force_diverge=a1_diverge)
            a1_diverge = False

            # Agent 2 
            a2_action = agent2.choose_action(obs, force_diverge=a2_diverge)
            a2_diverge = False

            # Execute Agent 1
            next_obs, reward_obj, done, info = env.step(a1_action)

            r = reward_obj.value
            r -= 0.05  # small movement cost
            
            # Anti-oscillation loop penalty
            if len(agent1.pos_history) == 4 and agent1.pos_history[0] == agent1.pos_history[2] and agent1.pos_history[1] == agent1.pos_history[3] and agent1.pos_history[0] != agent1.pos_history[1]:
                r -= 0.1
                
            # Idle penalty (-0.02)
            if a1_action.action_type not in ["pick", "drop"] and (a1_action.action_type != "move" or getattr(env.state_manager, 'idle_steps', {}).get("agent1", 0) > 0):
                r -= 0.02
                
            breakdown = info.get("reward_breakdown", {})
            if breakdown.get("collision", 0) < 0 or breakdown.get("agent_collision", 0) < 0:
                r -= 0.8   # strong collision penalty
                ep_collisions += 1
                a1_diverge = True
            if breakdown.get("pickup", 0) > 0:
                r += 1.5   
            if breakdown.get("delivery", 0) > 0:
                r += 4.0   
                ep_deliveries += 1
            if "coordination_efficiency" in info:
                ep_coordination = info["coordination_efficiency"]

            agent1.update(obs, a1_action, r, next_obs)
            total_reward += r

            obs = next_obs
            if done: break

            # Execute Agent 2
            next_obs, reward_obj, done, info = env.step(a2_action)
            r2 = reward_obj.value
            r2 -= 0.05
            
            if len(agent2.pos_history) == 4 and agent2.pos_history[0] == agent2.pos_history[2] and agent2.pos_history[1] == agent2.pos_history[3] and agent2.pos_history[0] != agent2.pos_history[1]:
                r2 -= 0.1
                
            if a2_action.action_type not in ["pick", "drop"] and (a2_action.action_type != "move" or getattr(env.state_manager, 'idle_steps', {}).get("agent2", 0) > 0):
                r2 -= 0.02
                
            breakdown2 = info.get("reward_breakdown", {})
            if breakdown2.get("collision", 0) < 0 or breakdown2.get("agent_collision", 0) < 0:
                r2 -= 0.8
                ep_collisions += 1
                a2_diverge = True
            if breakdown2.get("pickup", 0) > 0:
                r2 += 1.5
            if breakdown2.get("delivery", 0) > 0:
                r2 += 4.0
                ep_deliveries += 1
                
            agent2.update(obs, a2_action, r2, next_obs)
            total_reward += r2

            obs = next_obs
            if done: break
            
            if step % 20 == 0:
                print(f"[Ep {episode+1} Step {step}] A1 State: {agent1._get_state(obs)[:4]}... | Act: ({a1_action.action_type},{a1_action.direction}) | A2 Act: ({a2_action.action_type},{a2_action.direction}) | Reward: {r:.3f}")

        agent1.end_episode()
        agent2.end_episode()

        rewards_per_episode.append(total_reward)
        collision_rates.append(ep_collisions)
        delivery_rates.append(ep_deliveries)
        coordination_rates.append(ep_coordination)
        
        print(f"\n[Episode {episode+1:02d}]")
        print(f"Reward: {total_reward:.2f} | Collisions: {ep_collisions}")
        print(f"Epsilon: {agent1.epsilon:.2f}")

    print("\n=== TRAINING SUMMARY ===")
    print(f"Start Reward : {rewards_per_episode[0]:.2f}")
    print(f"Final Reward : {rewards_per_episode[-1]:.2f}")
    print(f"Improvement  : {rewards_per_episode[-1] - rewards_per_episode[0]:.2f}\n")

    # Save both Q-tables
    import pickle
    tables = {
        "agent1": agent1.q_values,
        "agent2": agent2.q_values
    }
    with open("q_table.pkl", "wb") as f:
        pickle.dump(tables, f)
    print("Saved Q-tables for both agents to q_table.pkl")

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
