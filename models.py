"""
Type-safe contracts for OpenEnv environment
Following OpenEnv course module-1 structure
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal


class Action(BaseModel):
    """Action that agent takes in environment"""
    decision: Literal["ESCALATE", "ANSWER"]
    response: Optional[str] = None


class Observation(BaseModel):
    """What agent sees from environment"""
    query: str
    context: str
    step_count: int
    task_id: str
    patient_id: int
    ground_truth: str
    last_action: Optional[str] = None
    last_reward: Optional[float] = None


class State(BaseModel):
    """Episode metadata (not seen by agent)"""
    episode_id: str
    step_count: int
    max_steps: int
    current_patient_id: Optional[int]
    current_task_id: str
    history: list
    episode_active: bool


class StepResult(BaseModel):
    """Result returned by reset() and step()"""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]