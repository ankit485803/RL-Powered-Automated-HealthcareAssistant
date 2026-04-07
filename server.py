
"""
FastAPI Server for OpenEnv Compatibility
Responsible: Shubham Kumar & Ankit Kumar
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from environment.healthcare_env import HealthcareEnvironment

app = FastAPI(title="RL-Powered Healthcare Assistant", description="OpenEnv Environment Server")

# Initialize environment
env = HealthcareEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "triage"


class StepRequest(BaseModel):
    response: str
    task_id: str = "triage"


class Action(BaseModel):
    decision: str
    response: Optional[str] = None


@app.get("/health")
async def health_check():
    """Liveness probe for OpenEnv compatibility"""
    return {"status": "ok"}


@app.post("/reset")
async def reset_episode(request: ResetRequest):
    """Start a new episode"""
    observation = env.reset(task_id=request.task_id)
    return observation


@app.post("/step")
async def step_action(request: StepRequest):
    """Submit an action and get next state, reward"""
    # Parse response into Action format
    # TODO: Implement action parsing and step logic
    return {"observation": {}, "reward": 0.0, "done": False, "info": {}}


@app.get("/state")
async def get_state():
    """Get current environment state"""
    return env.get_state()


@app.get("/tasks")
async def get_tasks():
    """Return available task catalogue"""
    return {
        "tasks": [
            {"id": "triage_easy", "name": "Emergency Detection", "difficulty": "easy"},
            {"id": "triage_medium", "name": "Severity Classification", "difficulty": "medium"},
            {"id": "triage_hard", "name": "Full Response Generation", "difficulty": "hard"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)