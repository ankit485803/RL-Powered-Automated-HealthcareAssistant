
# environment/healthcare_env.py (Skeleton)

"""
Healthcare Environment - OpenEnv Gymnasium-style interface
Responsible: Shubham Kumar
"""

import json
from typing import Tuple, Dict, Any, Optional


class HealthcareEnvironment:
    """
    OpenEnv-compatible Healthcare Triage Environment
    
    Methods:
        reset() -> Observation: Start new episode, return initial state
        step(action) -> (Observation, reward, done, info): Execute action
    """
    
    def __init__(self):
        """Initialize environment with patient query corpus"""
        self.current_episode = None
        self.step_count = 0
        self.max_steps = 5
        self.history = []
        
        # Patient corpus (to be populated by Ritu)
        self.patient_corpus = self._load_patient_corpus()
    
    def _load_patient_corpus(self) -> list:
        """Load patient queries from JSON or define inline"""
        # TODO: Ritu - Add 10-12 patient cases
        return [
            {
                "id": 1,
                "query": "I have chest pain and shortness of breath",
                "context": "45yo male, smoker",
                "ground_truth": "emergency",
                "requires_escalation": True
            },
            {
                "id": 2,
                "query": "I have a mild headache since yesterday",
                "context": "25yo female, no prior issues",
                "ground_truth": "non_emergency",
                "requires_escalation": False
            }
        ]
    
    def reset(self, task_id: str = "triage") -> Dict[str, Any]:
        """
        Start a new episode
        
        Args:
            task_id: Which task to run (easy/medium/hard)
        
        Returns:
            Observation dictionary with query, context, step_count
        """
        # TODO: Shubham - Implement reset logic
        self.step_count = 0
        self.history = []
        
        observation = {
            "query": "",
            "context": "",
            "step_count": self.step_count,
            "task_id": task_id
        }
        
        return observation
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment
        
        Args:
            action: {"decision": "ESCALATE" or "ANSWER", "response": str}
        
        Returns:
            (next_observation, reward, done, info)
        """
        # TODO: Shubham - Implement step logic
        # TODO: Ritu - Reward calculation will be called here
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        observation = {
            "query": "",
            "context": "",
            "step_count": self.step_count,
            "task_id": ""
        }
        
        reward = 0.0
        info = {}
        
        return observation, reward, done, info
    
    def get_state(self) -> Dict[str, Any]:
        """Return current environment state for debugging"""
        return {
            "step_count": self.step_count,
            "history": self.history,
            "current_episode": self.current_episode
        }