# 🏭 **Smart Warehouse: Multi-Agent AI that Learns to Coordinate**

## 🚀 **What We Built**
A **trainable multi-agent warehouse environment** where two AI agents must:

- 🤝 Coordinate under shared constraints  
- 🧭 Plan across long horizons (up to 300 steps)  
- 🌍 Act under uncertainty  
- 📈 Improve through real training  

---

## 🧠 **Why This Matters**
Most AI benchmarks test **single-agent reasoning**.

Real-world systems require:

> 🔥 **Multi-Agent Coordination + Long-Horizon Planning + Adaptation**

This project directly targets that gap.

---

## 🤖 **What the Agents Do**
Each agent:

- 👀 Observes environment (position, items, battery)
- 🎯 Takes actions (move, pick, deliver, charge)
- 📊 Learns from rewards

### 📦 Task Flow
Pick → Move → Deliver → Validate  

Hard tasks include **dependency chains**, forcing real planning.

---

## ⚡ **Multi-Agent Intelligence**
Agents must:

- ❌ Avoid collisions  
- ❌ Prevent duplicate work  
- ⚡ Share limited resources (charging station)  

### 📊 Key Signals:
- `coordination_efficiency`
- `intent_conflict`
- `agent_collisions`

---

## 📊 **Training Proof**
After training:

- 📈 Reward increases  
- 📉 Collisions decrease  
- 📦 Deliveries increase  

> ✅ Shows **real learning**, not scripted behavior

---

## 🔁 **Online RL (Real-Time Learning)**

- ⚡ Lightweight Q-learning  
- ⏱ Learns in ~50 episodes  
- 💾 Saves policy (`q_table.pkl`)  
- 🧠 Used in `/predict` API  

---

## 🌐 **Live System**

- ⚙️ FastAPI backend (OpenEnv compliant)  
- 🎮 Interactive UI dashboard  
- 📡 Real-time simulation + metrics  

> 👀 Judges can **see agents learning live**

---

## 🎯 **Theme Coverage (Hackathon Alignment)**

### 🤝 Theme #1 — Multi-Agent Interactions
- Cooperation + competition  
- Resource sharing (charger)  
- Conflict detection (`intent_conflict`)  
- Emergent coordination behavior  

### 🧭 Theme #2 — Long-Horizon Planning
- 300-step tasks  
- Multi-stage order completion  
- Dependency chains (order1 → order2 → order3)  

### 🌍 Theme #3 — World Modeling
- Partial observability  
- Dynamic obstacles  
- Stochastic item movement  
- Agents adapt to changing environment  

### 🔁 Theme #4 — Self-Improvement
- Online RL (Q-learning)  
- Adaptive curriculum  
- Measurable reward improvement  

---

## 🏁 **Key Takeaway**

👉 This environment enables AI to:

- Think beyond single-agent logic  
- Coordinate in real-world scenarios  
- Learn and improve over time  

---

## 🔗 **Links**

- 💻 GitHub  
  https://github.com/Alquama-Shaibli/MALH-Ware-Multi-Agent-Long-Horizon-Warehouse-Intelligence  

- 📓 Colab (Training Demo)  
  https://colab.research.google.com/drive/1W1NMqiOcWIAJK0XWdtSaVpZMq3NmD06p?usp=sharing  

- 🌐 Live Environment  
  *(Add Hugging Face Space link)*  

- 🎬 Demo Video  
  *(Add YouTube link)*  

---

## 🔥 **Final Line**

> 🚀 From single-agent reasoning → to real-world multi-agent intelligence
