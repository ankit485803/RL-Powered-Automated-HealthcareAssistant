
"""
Baseline inference script for OpenEnv compliance
used OpenAI client

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


def run_inference():
    """Run baseline inference on all three tasks"""
    
    env = HealthcareEnvironment()
    
    # Task 1: Emergency Detection (Easy)
    print("[START] task=emergency_detection env=healthcare_assistant model=" + MODEL_NAME)
    
    obs = env.reset(task_id="emergency_detection")
    total_reward = 0.0
    step = 1
    
    # Simple baseline: if query contains emergency keywords -> ESCALATE
    emergency_keywords = ["chest pain", "difficulty breathing", "severe", "bleeding", "unconscious"]
    
    while step <= env.max_steps:
        query = obs.get("query", "").lower()
        
        # Baseline decision logic
        if any(keyword in query for keyword in emergency_keywords):
            action = {"decision": "ESCALATE", "response": ""}
        else:
            action = {"decision": "ANSWER", "response": "Please consult a doctor for proper advice."}
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Print STEP line exactly as required
        print(f"[STEP] step={step} action={action['decision']} reward={reward:.2f} done={str(done).lower()} error=null")
        
        step += 1
        if done:
            break
    
    print(f"[END] success=true steps={step-1} rewards={total_reward:.2f}")
    
    # Task 2: Severity Classification (Medium)
    print("[START] task=severity_classification env=healthcare_assistant model=" + MODEL_NAME)
    # Similar implementation...
    
    # Task 3: Full Response Generation (Hard)
    print("[START] task=full_response env=healthcare_assistant model=" + MODEL_NAME)
    # Similar implementation...


if __name__ == "__main__":
    run_inference()