# RL-Powered Automated Healthcare Assistant

## Abstract
(Write 150 words: problem, approach, MDP formulation, key reward components)

## 1. Introduction
- Healthcare triage gap
- Why RL over rule-based systems
- OpenEnv as framework

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
R = w1 × Safety + w2 × Quality + w3 × Efficiency

Safety = +10 if correct escalate, -20 if missed emergency
Quality = LLM-as-judge score (0-5)
Efficiency = -0.1 per step

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
├── .gitignore                # Exclusions
└── README.md                 # Documentation

```



## 8. References
(3-5 citations)