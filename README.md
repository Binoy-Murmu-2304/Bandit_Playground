# 🎰 Bandit Playground

An interactive exploration of the **exploration-exploitation dilemma** —
the core challenge of multi-armed bandits and reinforcement learning.

Built to internalise Sutton & Barto, Ch. 2 — then apply the insights to
real-world systems like the [Self-Healing MLOps Engine](#-connection-to-real-world-systems).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)

---

## ✨ Features

| Page | What It Does |
|------|-------------|
| **🔬 Live Simulation** | Pull arms step-by-step. Watch Q-values update in real time. Feel the tradeoff. |
| **📊 Comparison** | Run all 5 algorithms on the same bandit. Overlay curves. See who wins. |
| **🌍 Non-Stationary** | The world shifts mid-game. Watch sample-average fail and constant-α recover. |
| **📝 Findings** | Key insights with evidence. Ready for interviews and viva prep. |

## 🤖 Algorithms Implemented

| Algorithm | Strategy | Key Idea |
|-----------|----------|----------|
| **Greedy** | Pure exploitation | Always pick the current best |
| **ε-greedy** | Random exploration | Explore with probability ε |
| **Optimistic Init** | Curiosity-driven | Start inflated, reality corrects |
| **UCB** | Uncertainty-driven | Bonus for under-tried arms |
| **Thompson Sampling** | Bayesian | Sample from posterior distributions |

## 📊 Key Findings

| # | Finding | Numbers |
|---|---------|---------|
| 1 | Greedy gets stuck permanently | 35.8% optimal, never improves |
| 2 | The best ε depends on your budget | ε=0.1 wins at 1000 pulls; ε=0.01 at 10,000 |
| 3 | Constant α adapts to drift; 1/N doesn't | ~50-step recovery vs. slow crawl |
| 4 | UCB is the smartest explorer | 85.7% optimal — best overall |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/bandit-playground.git
cd bandit-playground