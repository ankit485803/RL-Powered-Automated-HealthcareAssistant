
# 🏥 RL-Powered Automated Healthcare Assistant

### Team Vector | Meta PyTorch OpenEnv Hackathon 2026

---

## 📌 Overview

An **OpenEnv-compatible Reinforcement Learning environment** where an AI agent learns optimal clinical triage decisions — choosing between **ESCALATE** (alert healthcare professional) or **ANSWER** (generate LLM response) for patient queries.

The agent learns through delayed rewards, balancing:
- ✅ **Patient Safety** (never miss emergencies)
- ✅ **Resource Efficiency** (avoid false alarms)
- ✅ **Response Quality** (accurate, helpful answers)

---

## 🎯 Problem Statement

| Challenge | Our Solution |
|-----------|--------------|
| Medical queries need triage | RL agent learns escalation policy |
| LLMs can miss emergencies | Hierarchical reward structure penalizes missed emergencies severely (-20) |
| False alarms waste resources | False alarm penalty (-5) |
| Response quality matters | LLM-as-judge evaluates answer quality (0-5) |

---

## 🏗️ Architecture
