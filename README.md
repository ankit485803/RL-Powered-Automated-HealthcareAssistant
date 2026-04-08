
# RL-Powered Automated Healthcare Assistant  
**Team Vector**  
- Shubham Kumar  
- Ankit Kumar (Lead)  
- Ritu Ranjan  

---

## Abstract
Healthcare triage systems face a critical gap: rule-based algorithms fail on nuanced cases, while LLMs lack safety and escalation discipline. We propose an OpenEnv-compatible Reinforcement Learning environment for clinical triage. The problem is modeled as a Markov Decision Process (MDP) with state (patient query + context), actions (ESCALATE or ANSWER), and a reward function balancing safety, response quality, and efficiency. The system penalizes missed emergencies, discourages false alarms, and rewards high-quality responses. Three difficulty levels are included: emergency detection, severity classification, and full response generation. The environment includes a grader, Docker setup, and baseline inference pipeline for reproducible evaluation.

---

## 1. Introduction

### Healthcare Triage Gap
Healthcare systems handle massive volumes of patient queries daily. Identifying true emergencies among them is critical.  
- Missed emergencies → high risk  
- Over-escalation → resource wastage  

---

### Why Reinforcement Learning?
- Rule-based systems → rigid and brittle  
- LLMs → flexible but unsafe  
- RL → learns optimal decisions via feedback  

RL enables:
- Adaptive learning from interactions  
- Better safety–efficiency trade-offs  
- Continuous improvement over time  

---

### OpenEnv Framework
We use OpenEnv to build a standardized RL environment:
- `reset()` → initialize state  
- `step(action)` → transition + reward  
- `state()` → current observation  

Supports:
- Reproducible evaluation  
- Containerized deployment (Docker)  
- Integration with Hugging Face Spaces  

---

## 1.1 Problem Statement
The task is sequential decision-making under uncertainty.

Given:
- Patient query  
- Limited clinical context  

Agent must decide:
- **ESCALATE** → refer to human expert  
- **ANSWER** → generate safe response  

## 1.2 Objectives
- **Patient Safety** → avoid missed emergencies  
- **Resource Efficiency** → reduce false escalations  
- **Response Quality** → ensure accurate guidance  

The environment provides:
- Dense reward signals  
- Multi-level difficulty tasks  
- Standardized evaluation setup  

## 2. Environment Design
### 2.1 MDP Formulation
| Component | Definition |
|-----------|------------|
| State | Current patient query + history |
| Observation | {query_text, patient_context, step_count} |
| Action | {decision: ESCALATE or ANSWER, response: str} |
| Reward | Hierarchical (safety + quality) |
| Horizon | 5 steps per episode |
| Terminal | Emergency detected OR max steps reached |

### 2.2 Patient Query Corpus
(Table of 10-12 test cases: symptoms, ground truth escalation needed, expected action)

## 3. Tasks
### Task 1 — Emergency Detection (Easy)
- Action: binary (ESCALATE or ANSWER)
- Reward: +10 correct escalate, -20 missed emergency

### Task 2 — Severity Classification (Medium)
- Action: mild / moderate / severe / emergency
- Partial credit for adjacent levels

### Task 3 — Full Response Generation (Hard)
- Action: ESCALATE or ANSWER with LLM response
- Multi-component reward


## 4. Reward Function

The reward function is defined as:

**R = w₁ × Safety + w₂ × Quality + w₃ × Efficiency**

### Components

- **Safety**
  - 1.0 → Correct escalation  
  - 0.0 → Missed emergency  
  - 0.2 → False escalation  

- **Quality**
  - Normalized LLM-based score ∈ [0.3, 1.0]  
  - Reflects accuracy, safety, and completeness  

- **Efficiency**
  - Time penalty ≤ 0.3  
  - Encourages faster responses  

### Note
The final reward is normalized to the range **[0.0, 1.0]** to ensure compatibility with OpenEnv.


## 5. Server API
| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Liveness check |
| /reset | POST | Start new episode |
| /step | POST | Submit action |
| /state | GET | Current environment state |

## 6. Setup and Deployment
### Local Installation
(commands)

### Docker Deployment
(commands)

## 7. Project Structure  
*(Tree with file purposes)*

```text
metaHFpytorch_openEnvHackathon/
│
├── environment/
│   ├── __init__.py           # Module exports
│   ├── healthcare_env.py     # OpenEnv core: reset(), step(), get_state()
│   └── reward.py             # Hierarchical reward calculation
│
├── agent/
│   ├── __init__.py           # Module exports
│   ├── rl_agent.py           # PPO/GRPO policy network
│   └── train.py              # Training loop
│
├── llm/
│   ├── __init__.py           # Module exports
│   └── gemini_handler.py     # Gemini API wrapper
│
├── tests/
│   └── test_reward.py        # Unit tests for reward function
│
├── docker/
│   └── Dockerfile            # Container definition
│
├── docs/
│   └── dataFlow.txt          # Data flow diagram
│
├── server.py                 # FastAPI endpoints
├── requirements.txt          # Dependencies
├── inference.py              # prediction logic file  
├── openenv.py                # env setup 
├── .gitignore                # Exclusions ignored files  
└── README.md                 # Documentation

```


## 8. RL Architecture & Algorithm

### 8.1 Agent-Environment Loop

The system implements a standard Reinforcement Learning loop:

1. **Observe** – Environment sends patient query + context to agent  
2. **Act** – Agent chooses ESCALATE or ANSWER  
3. **Reward** – Environment returns hierarchical reward (safety + quality + efficiency)  
4. **Learn** – Agent updates policy using PPO  
5. **Repeat** – Until episode ends (max 5 steps or emergency detected)  

### 8.2 Algorithm: Proximal Policy Optimization (PPO)

| Property            | Value                                                         |
|---------------------|---------------------------------------------------------------|
| **Type**            | Actor-Critic                                                  |
| **Why chosen**      | Stable updates, sample efficient, handles continuous/discrete actions |
| **Alternatives rejected** | Q-Learning (unstable for long horizons), REINFORCE (high variance) |

**PPO Clipping Mechanism:**

\[
L(\theta) = \min\big(r(\theta) A, \ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \big)
\]

where \( r(\theta) \) = probability ratio between new and old policy, \( A \) = advantage estimate.

### 8.3 Policy Network Architecture

Input (512-dim embedding)  
↓  
Dense(256) + ReLU  
↓  
Dense(256) + ReLU  
↓  
Dense(2) + SoftMax  
↓  
Output: [P(ESCALATE), P(ANSWER)]  

### 8.4 Training Configuration

| Parameter           | Value  |
|---------------------|--------|
| Learning rate       | 3e-4   |
| Discount factor (γ) | 0.99   |
| GAE λ               | 0.95   |
| PPO clip range (ε)  | 0.2    |
| Epochs per update   | 4      |
| Batch size          | 64     |


## 9. References
(3-5 citations)