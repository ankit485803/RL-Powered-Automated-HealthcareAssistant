
"""
OpenEnv Client - What training code imports
Following OpenEnv course module-1 structure
"""

import asyncio
import httpx
from typing import Dict, Any, Optional

from models import Observation, StepResult, State


class HealthcareEnvClient:
    """
    Client for Healthcare Triage Environment
    Communicates with server via HTTP/WebSocket
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self._current_task_id = "emergency_detection"
    
    def reset(self, task_id: str = "emergency_detection") -> StepResult:
        """Start new episode"""
        response = self.client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
        )
        response.raise_for_status()
        data = response.json()
        
        self._current_task_id = task_id
        
        return StepResult(
            observation=Observation(**data.get("observation", {})),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {})
        )
    
    def step(self, action: Dict[str, Any]) -> StepResult:
        """Execute action"""
        response = self.client.post(
            f"{self.base_url}/step",
            json={
                "decision": action.get("decision", "ANSWER"),
                "response": action.get("response", "")
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return StepResult(
            observation=Observation(**data.get("observation", {})),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {})
        )
    
    def state(self) -> State:
        """Get current episode state"""
        response = self.client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return State(**response.json())
    
    def close(self):
        """Close client session"""
        self.client.close()


# Async version for WebSocket (optional)
class AsyncHealthcareEnvClient:
    """Async client for WebSocket communication"""
    
    def __init__(self, base_url: str = "ws://localhost:8000/ws"):
        self.base_url = base_url
        self._ws = None
    
    async def connect(self):
        import websockets
        self._ws = await websockets.connect(self.base_url)
    
    async def reset(self, task_id: str = "emergency_detection") -> StepResult:
        await self._ws.send(f'{{"type": "reset", "task_id": "{task_id}"}}')
        response = await self._ws.recv()
        # Parse response...
        return StepResult(observation={}, reward=0.0, done=False, info={})
    
    async def step(self, action: Dict[str, Any]) -> StepResult:
        await self._ws.send(f'{{"type": "step", "action": {action}}}')
        response = await self._ws.recv()
        return StepResult(observation={}, reward=0.0, done=False, info={})
    
    async def close(self):
        if self._ws:
            await self._ws.close()