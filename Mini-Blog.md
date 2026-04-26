🏭 Smart Warehouse: Multi-Agent AI that Learns to Coordinate
🚀 What We Built

A trainable multi-agent warehouse environment where two AI agents must:

Coordinate under shared constraints
Plan across long horizons (up to 300 steps)
Act under uncertainty (dynamic obstacles, partial observability)
Improve through real training
🧠 Why This Matters

Most AI benchmarks test single-agent reasoning.
Real-world systems (warehouses, fleets, robotics) require:

👉 multi-agent coordination + long-term planning

This environment directly targets that gap.

🤖 Agent Behavior

Each agent continuously:

Observes state (position, items, battery)
Chooses actions (move, pick, deliver, charge)
Learns from reward

Task pipeline:

👉 Pick → Move → Deliver → Validate

Hard tasks include dependency chains, forcing true planning.

⚡ Multi-Agent Intelligence

Agents must:

Avoid collisions
Prevent duplicate work
Share resources (charging station)

We measure:

coordination_efficiency
intent_conflict
collisions
📊 Training Proof

After training:

📈 Reward increases
📉 Collisions decrease
📦 Deliveries increase

👉 Clear evidence of learning, not scripting

🔁 Online RL (Real Learning)

Implemented lightweight Q-learning:

Learns in ~50 episodes
Policy saved and reused
Powers real-time decisions via API
🌐 Live System

Includes:

FastAPI environment (OpenEnv compliant)
Real-time UI dashboard
Live metrics + visualization
RL-driven action prediction
🏁 Key Takeaway

👉 This environment enables training AI that can:

Coordinate with other agents
Plan over long tasks
Adapt to dynamic worlds
🔗 Links
GitHub: https://github.com/Alquama-Shaibli/MALH-Ware-Multi-Agent-Long-Horizon-Warehouse-Intelligence
Colab: https://colab.research.google.com/drive/1W1NMqiOcWIAJK0XWdtSaVpZMq3NmD06p?usp=sharing
Demo: (add your links)