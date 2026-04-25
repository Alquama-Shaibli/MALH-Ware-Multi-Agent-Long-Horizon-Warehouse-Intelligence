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
        self.epsilon_decay = 0.90
        self.min_epsilon = 0.05
        self.lr = 0.2
        self.gamma = 0.95

    def _get_state(self, obs):
        if self.agent_id not in obs.robots:
            return (0, 0, False)
        robot = obs.robots[self.agent_id]
        is_carrying = len(robot.carrying) > 0
        rx, ry = robot.pos
        
        # Simplified spatial awareness: direction to target
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
        sdx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        sdy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        
        # Check for adjacent obstacles
        obstacle_nearby = False
        if hasattr(obs, 'obstacles') and obs.obstacles:
            for obs_pos in obs.obstacles:
                # obs_pos could be a dict/list/object
                ox, oy = obs_pos if isinstance(obs_pos, (list, tuple)) else (obs_pos['pos'] if isinstance(obs_pos, dict) and 'pos' in obs_pos else obs_pos.pos)
                if abs(ox - rx) <= 1 and abs(oy - ry) <= 1:
                    obstacle_nearby = True
                    break

        return (sdx, sdy, is_carrying, obstacle_nearby)

    def choose_action(self, obs):
        state = self._get_state(obs)
        if state not in self.q_values:
            self.q_values[state] = {a: 0.0 for a in self.actions}

        if random.random() < self.epsilon:
            action_tuple = random.choice(self.actions)
        else:
            action_tuple = max(self.actions, key=lambda a: self.q_values[state][a])
            
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
    
    print("Starting Online RL Training (50 Episodes)...")

    for episode in range(50):
        # Force easy task to ensure it converges in 50 episodes
        obs = env.reset(task="easy")
        total_reward = 0

        for step in range(100):
            # Agent 1 Learning
            a1_action = agent1.choose_action(obs)
            # Agent 2 acts randomly
            a2_action = Action(agent_id="agent2", action_type="move", direction=random.choice(["up", "down", "left", "right"]))
            
            # Since step takes a single action in this environment loop, we interleave
            next_obs, reward_obj, done, _ = env.step(a1_action)
            agent1.update(obs, a1_action, reward_obj.value, next_obs)
            total_reward += reward_obj.value
            obs = next_obs
            
            if done: break
            
            # Agent 2 step
            next_obs, reward_obj, done, _ = env.step(a2_action)
            obs = next_obs
            
            if done: break

        agent1.end_episode()
        agent2.end_episode()

        rewards_per_episode.append(total_reward)
        print(f"[Episode {episode+1:02d}] Reward: {total_reward:.2f} | Epsilon: {agent1.epsilon:.2f}")

    return rewards_per_episode

# -----------------------------
# Plot Results
# -----------------------------
def plot_results(rewards):
    plt.figure(figsize=(8,5))
    
    # Smooth the curve for better visual impact
    smoothed_rewards = [np.mean(rewards[max(0, i-5):i+1]) for i in range(len(rewards))]
    
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    plt.plot(smoothed_rewards, color='blue', linewidth=2, marker='o', label='Smoothed Trend')
    
    plt.title("Online RL Learning Curve (Tabular Q-Learning)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig("online_rl_curve.png", dpi=150)
    print("Online RL completed - learning curve saved as 'online_rl_curve.png'")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    rewards = run_online_rl()
    plot_results(rewards)
