---
title: Smart Warehouse Multi-Agent Intelligence Environment (OpenEnv)
emoji: 🏭
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - multi-agent
  - long-horizon
  - reinforcement-learning
  - fastapi
---

# Smart Warehouse Multi-Agent Intelligence Environment

> **TL;DR:** A Multi-Agent warehouse environment where two robots cooperate to fulfill orders. Features a novel **"Fleet AI"** meta-agent that monitors intent and intervenes to prevent collisions using Theory of Mind. Includes a complete, runnable TRL-based SFT pipeline proving LLM policy adaptation (distilgpt2 loss: 3.93 → 0.14).

![Training Curve](training_curve.png)

## 🔗 Quick Links 

- **Hugging Face Space (Runnable Env)**: [Smart Warehouse Space](https://huggingface.co/spaces/AlquamaShaibli/MALH-Ware-Multi-Agent-Long-Horizon-Warehouse-Intelligence)
- **GitHub Repository**: [MALH-Ware-Multi-Agent-Long-Horizon-Warehouse-Intelligence](https://github.com/Alquama-Shaibli/MALH-Ware-Multi-Agent-Long-Horizon-Warehouse-Intelligence)
- **Colab Notebook (Training Pipeline)**: [Open in Colab](https://colab.research.google.com/drive/1W1NMqiOcWIAJK0XWdtSaVpZMq3NmD06p)
- **Mini-blog (Hugging Face post)**: *(ADD LINK HERE)*
- **Demo Video (< 2 mins)**: *(ADD LINK HERE)*

## ✨ Key Features (OpenEnv Themes)

- 🤖 **Multi-Agent Interactions**: Robots share resources (charger), coordinate, and compete. Includes *Intent Conflict Detection* and *Theory of Mind*.
- 🎯 **Long-Horizon Planning**: Chained order dependencies requiring up to 300 steps and 12 sequential subgoals to complete.
- 🌍 **World Modeling**: Partial observability (radius=2) with stochastic item relocation and dynamic moving obstacles.
- 🚀 **Self-Improvement**: Features an adaptive curriculum and a complete SFT pipeline (`train_llm.py`) proving real policy adaptation.

## 📊 Visual Proof of Self-Improvement

![Self-Improvement Curve](self_improvement_curve.png)
![SFT Training Curve](sft_training_curve.png)

*Curriculum Learning Note: Initial negative rewards on "Hard" tasks in RL plots represent the agent safely exploring complex penalties before mastering the task. The SFT pipeline above proves final capability acquisition.*

## 🚀 Quickstart

```bash
# 1. Install environment
pip install -e .

# 2. Start API server (OpenEnv multi-mode compatible)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Validate Submission
python train_llm.py --validate
```

## 📚 Technical Details
For an in-depth breakdown of the Fleet AI architecture, the 19-component reward system, and multi-agent coordination mechanics, please read [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).
