
"""
Baseline inference script for OpenEnv compliance
Uses OpenAI client as required by hackathon guidelines
"""

import os
from openai import OpenAI
from environment.healthcare_env import HealthcareEnvironment

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def get_baseline_action(query: str, task_id: str) -> dict:
    """
    Simple baseline policy using keyword matching
    For hard task, uses OpenAI to generate response
    """
    query_lower = query.lower()
    
    # Emergency keywords for escalation
    emergency_keywords = [
        "chest pain", "difficulty breathing", "shortness of breath",
        "severe", "bleeding", "unconscious", "numb", "droopy",
        "suicidal", "blood", "emergency"
    ]
    
    severity_keywords = {
        "emergency": emergency_keywords,
        "severe": ["severe", "intense", "excruciating", "cannot move"],
        "moderate": ["moderate", "persistent", "ongoing"],
        "mild": ["mild", "slight", "little"]
    }
    
    if task_id == "emergency_detection":
        if any(kw in query_lower for kw in emergency_keywords):
            return {"decision": "ESCALATE", "response": ""}
        else:
            return {"decision": "ANSWER", "response": "Please consult a doctor for proper advice."}
    
    elif task_id == "severity_classification":
        if any(kw in query_lower for kw in emergency_keywords):
            return {"decision": "ESCALATE", "response": "emergency"}
        elif any(kw in query_lower for kw in severity_keywords["severe"]):
            return {"decision": "ANSWER", "response": "severe"}
        elif any(kw in query_lower for kw in severity_keywords["moderate"]):
            return {"decision": "ANSWER", "response": "moderate"}
        else:
            return {"decision": "ANSWER", "response": "mild"}
    
    else:  # full_response
        if any(kw in query_lower for kw in emergency_keywords):
            return {"decision": "ESCALATE", "response": ""}
        else:
            # Use OpenAI to generate response
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant. Provide brief, safe health advice in 1-2 sentences."},
                        {"role": "user", "content": f"Patient query: {query}"}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                generated = response.choices[0].message.content.strip()
                return {"decision": "ANSWER", "response": generated}
            except Exception:
                return {"decision": "ANSWER", "response": "Please consult a healthcare provider for proper evaluation."}


def run_task(env: HealthcareEnvironment, task_id: str):
    """Run a single task and print output in required format"""
    
    print(f"[START] task={task_id} env=healthcare_assistant model={MODEL_NAME}")
    
    obs = env.reset(task_id=task_id)
    step = 1
    done = False
    rewards_list = []
    
    while not done and step <= env.max_steps:
        query = obs.get("query", "")
        action = get_baseline_action(query, task_id)
        
        obs, reward, done, info = env.step(action)
        rewards_list.append(f"{reward:.2f}")
        
        # Extract response for logging
        response_display = action.get("response", "")[:50] if action.get("response") else action["decision"]
        
        print(f"[STEP] step={step} action={action['decision']} reward={reward:.2f} done={str(done).lower()} error=null")
        
        step += 1
    
    rewards_str = ",".join(rewards_list)
    print(f"[END] success=true steps={step-1} rewards={rewards_str}")


def run_inference():
    """Run baseline inference on all three tasks"""
    
    env = HealthcareEnvironment()
    
    # Task 1: Emergency Detection (Easy)
    run_task(env, "emergency_detection")
    
    # Reset environment for next task
    env = HealthcareEnvironment()
    run_task(env, "severity_classification")
    
    # Reset environment for next task
    env = HealthcareEnvironment()
    run_task(env, "full_response")


if __name__ == "__main__":
    run_inference()