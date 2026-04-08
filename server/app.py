

"""
FastAPI Server for OpenEnv Compliance
Responsible: Shubham Kumar & Ankit Kumar
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from environment.healthcare_env import HealthcareEnvironment

# Create FastAPI app
app = FastAPI(
    title="RL-Powered Healthcare Assistant",
    description="OpenEnv Environment Server for Healthcare Triage",
    version="1.0.0"
)

# Global environment instance (each request gets new session in production)
_env_instance = None

def get_environment():
    """Get or create environment instance"""
    global _env_instance
    if _env_instance is None:
        _env_instance = HealthcareEnvironment()
    return _env_instance


class ResetRequest(BaseModel):
    task_id: str = "emergency_detection"


class StepRequest(BaseModel):
    response: str
    task_id: str = "emergency_detection"


class ActionRequest(BaseModel):
    decision: str
    response: Optional[str] = None


@app.get("/health")
async def health_check():
    """Liveness probe for OpenEnv compatibility"""
    return {"status": "ok"}


@app.post("/reset")
async def reset_episode(request: ResetRequest):
    """Start a new episode"""
    env = get_environment()
    observation = env.reset(task_id=request.task_id)
    return observation


@app.post("/step")
async def step_action(request: StepRequest):
    """
    Submit an action and get next state, reward
    
    Expected action format: {"decision": "ESCALATE" or "ANSWER", "response": str}
    """
    env = get_environment()
    
    # Parse the response string to extract decision
    # For now, assume simple format or use a simple rule
    response_lower = request.response.lower()
    
    # Simple decision logic for baseline
    emergency_keywords = ["chest pain", "breath", "numb", "blood", "severe", "suicidal", "emergency"]
    
    if any(keyword in response_lower for keyword in emergency_keywords):
        decision = "ESCALATE"
    else:
        decision = "ANSWER"
    
    action = {
        "decision": decision,
        "response": request.response if decision == "ANSWER" else ""
    }
    
    observation, reward, done, info = env.step(action)
    
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
async def get_state():
    """Get current environment state"""
    env = get_environment()
    return env.get_state()


@app.get("/tasks")
async def get_tasks():
    """Return available task catalogue"""
    return {
        "tasks": [
            {"id": "emergency_detection", "name": "Emergency Detection", "difficulty": "easy"},
            {"id": "severity_classification", "name": "Severity Classification", "difficulty": "medium"},
            {"id": "full_response", "name": "Full Response Generation", "difficulty": "hard"}
        ]
    }


def main():
    """Entry point for uvicorn server"""
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()