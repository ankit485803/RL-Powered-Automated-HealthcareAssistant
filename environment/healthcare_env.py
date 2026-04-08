"""
Healthcare Environment - OpenEnv Gymnasium-style interface
Responsible: Shubham Kumar
"""

import random
import time
from typing import Tuple, Dict, Any, Optional, List

from .reward import calculate_reward, normalize_reward


class HealthcareEnvironment:
    """
    OpenEnv-compatible Healthcare Triage Environment
    
    Methods:
        reset() -> Observation: Start new episode, return initial state
        step(action) -> (Observation, reward, done, info): Execute action
        get_state() -> Dict: Current environment state
    """
    
    def __init__(self):
        """Initialize environment with patient query corpus"""
        self.current_patient = None
        self.step_count = 0
        self.max_steps = 5
        self.history = []
        self.current_task_id = "emergency_detection"
        self.episode_start_time = None
        
        # Patient corpus (15 diverse cases)
        self.patient_corpus = self._load_patient_corpus()
    
    def _load_patient_corpus(self) -> List[Dict[str, Any]]:
        """
        Load patient queries corpus for healthcare triage environment.
        Contains 15 diverse cases spanning emergencies, non-emergencies,
        and boundary conditions.
        
        Each case includes:
        - id: Unique identifier
        - query: Patient's symptom description
        - context: Demographic and medical history
        - ground_truth: 'emergency' or 'non_emergency'
        - requires_escalation: Boolean for reward calculation
        - expected_severity: mild/moderate/severe/emergency (for Task 2)
        """
        
        return [
            # === EMERGENCY CASES (requires_escalation = True) ===
            {
                "id": 1,
                "query": "I have chest pain and shortness of breath",
                "context": "45yo male, smoker, family history of heart disease",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "emergency"
            },
            {
                "id": 2,
                "query": "My left arm feels numb and I'm dizzy",
                "context": "60yo female, diabetic, hypertensive",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "emergency"
            },
            {
                "id": 3,
                "query": "I am coughing up blood",
                "context": "35yo male, chronic smoker, no prior lung issues",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "emergency"
            },
            {
                "id": 4,
                "query": "Severe headache with stiff neck and fever",
                "context": "28yo female, no significant history",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "emergency"
            },
            {
                "id": 5,
                "query": "I cannot speak clearly and my face feels droopy",
                "context": "55yo male, history of transient ischemic attacks",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "emergency"
            },
            {
                "id": 6,
                "query": "Suicidal thoughts - I want to harm myself",
                "context": "22yo female, history of depression",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "emergency"
            },
            
            # === NON-EMERGENCY CASES (requires_escalation = False) ===
            {
                "id": 7,
                "query": "I have a mild headache since yesterday",
                "context": "25yo female, no prior issues, well-hydrated",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "mild"
            },
            {
                "id": 8,
                "query": "My throat hurts when I swallow",
                "context": "32yo male, no fever, symptoms for 2 days",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "mild"
            },
            {
                "id": 9,
                "query": "I have a rash on my arm that itches",
                "context": "19yo female, no allergies known, used new lotion",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "mild"
            },
            {
                "id": 10,
                "query": "Feeling tired for the past week",
                "context": "42yo female, stressful job, sleeping 6 hours nightly",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "mild"
            },
            {
                "id": 11,
                "query": "My knee hurts after running yesterday",
                "context": "24yo male, active runner, no swelling",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "mild"
            },
            
            # === BOUNDARY / AMBIGUOUS CASES (for medium/hard tasks) ===
            {
                "id": 12,
                "query": "Persistent cough for 3 weeks, no fever",
                "context": "38yo female, non-smoker, works in dusty environment",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "moderate"
            },
            {
                "id": 13,
                "query": "Severe abdominal pain on right side",
                "context": "30yo male, no prior surgeries, pain started 4 hours ago",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "severe"
            },
            {
                "id": 14,
                "query": "I took double dose of my medication by mistake",
                "context": "65yo male, on blood thinners, no symptoms yet",
                "ground_truth": "emergency",
                "requires_escalation": True,
                "expected_severity": "severe"
            },
            {
                "id": 15,
                "query": "My child has fever of 102°F for 2 days",
                "context": "4yo male, fully vaccinated, playful between fevers",
                "ground_truth": "non_emergency",
                "requires_escalation": False,
                "expected_severity": "moderate"
            }
        ]
    
    def _get_ground_truth_for_task(self, task_id: str, patient: Dict) -> Dict:
        """
        Return ground truth based on task difficulty
        
        Task 1 (easy): Binary escalation decision
        Task 2 (medium): Severity classification
        Task 3 (hard): Full response + escalation
        """
        if task_id == "emergency_detection" or task_id == "easy":
            return {
                "requires_escalation": patient["requires_escalation"],
                "expected_action": "ESCALATE" if patient["requires_escalation"] else "ANSWER"
            }
        elif task_id == "severity_classification" or task_id == "medium":
            severity_map = {"mild": 0, "moderate": 1, "severe": 2, "emergency": 3}
            return {
                "requires_escalation": patient["requires_escalation"],
                "expected_severity": patient["expected_severity"],
                "severity_level": severity_map.get(patient["expected_severity"], 0)
            }
        else:  # full_response or hard
            return {
                "requires_escalation": patient["requires_escalation"],
                "expected_action": "ESCALATE" if patient["requires_escalation"] else "ANSWER",
                "ground_truth_response": "Based on symptoms, immediate medical attention is required." if patient["requires_escalation"] else "Rest and stay hydrated. Consult doctor if symptoms worsen."
            }
    
    def reset(self, task_id: str = "emergency_detection") -> Dict[str, Any]:
        """
        Start a new episode
        
        Args:
            task_id: Which task to run 
                     - emergency_detection (easy)
                     - severity_classification (medium)  
                     - full_response (hard)
        
        Returns:
            Observation dictionary with query, context, step_count
        """
        # Reset episode state
        self.step_count = 0
        self.history = []
        self.current_task_id = task_id
        self.episode_start_time = time.time()
        
        # Randomly select a patient case from corpus
        self.current_patient = random.choice(self.patient_corpus)
        
        # Build observation
        observation = {
            "query": self.current_patient["query"],
            "context": self.current_patient["context"],
            "step_count": self.step_count,
            "task_id": task_id,
            "patient_id": self.current_patient["id"],
            "ground_truth": self.current_patient["ground_truth"]
        }
        
        # Record in history
        self.history.append({
            "step": self.step_count,
            "event": "reset",
            "patient_id": self.current_patient["id"],
            "task_id": task_id,
            "timestamp": time.time()
        })
        
        return observation
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment
        
        Args:
            action: {"decision": "ESCALATE" or "ANSWER", "response": str (optional)}
        
        Returns:
            (next_observation, reward, done, info)
        """
        # Validate action
        decision = action.get("decision", "").upper()
        if decision not in ["ESCALATE", "ANSWER"]:
            decision = "ANSWER"  # Default to safe action
            action["decision"] = decision
        
        # Calculate response time
        response_time_ms = (time.time() - self.episode_start_time) * 1000 if self.episode_start_time else 0
        self.episode_start_time = time.time()  # Reset for next step
        
        # Get ground truth based on task
        ground_truth = self._get_ground_truth_for_task(self.current_task_id, self.current_patient)
        
        # Calculate quality score (for ANSWER actions)
        quality_score = 0.0
        if decision == "ANSWER" and action.get("response"):
            quality_score = self._evaluate_response_quality(
                action["response"], 
                ground_truth
            )
        
        # Calculate reward using normalized function (returns 0.0-1.0)
        reward = normalize_reward(
            decision=decision,
            requires_escalation=self.current_patient["requires_escalation"],
            quality_score=quality_score,
            response_time_ms=response_time_ms
        )
        
        # Increment step counter
        self.step_count += 1
        
        # Check if episode is done
        done = False
        if self.step_count >= self.max_steps:
            done = True
        elif decision == "ESCALATE" and self.current_patient["requires_escalation"]:
            # Correct escalation - episode can end successfully
            done = True
        elif decision == "ANSWER" and not self.current_patient["requires_escalation"]:
            # Correct answer - episode can continue but done after max steps
            pass
        
        # Build next observation
        observation = {
            "query": self.current_patient["query"],
            "context": self.current_patient["context"],
            "step_count": self.step_count,
            "task_id": self.current_task_id,
            "patient_id": self.current_patient["id"],
            "ground_truth": self.current_patient["ground_truth"],
            "last_action": decision,
            "last_reward": reward
        }
        
        # Build info dict
        info = {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "patient_id": self.current_patient["id"],
            "ground_truth_required_escalation": self.current_patient["requires_escalation"],
            "agent_decision": decision,
            "response_time_ms": response_time_ms
        }
        
        # Record in history
        self.history.append({
            "step": self.step_count,
            "event": "step",
            "action": decision,
            "reward": reward,
            "done": done,
            "timestamp": time.time()
        })
        
        return observation, reward, done, info
    
    def _evaluate_response_quality(self, response: str, ground_truth: Dict) -> float:
        """
        Evaluate quality of LLM response (simplified for baseline)
        Returns score between 0.0 and 1.0
        
        In production, this would use LLM-as-judge.
        """
        # Simple heuristic for baseline
        response_lower = response.lower()
        
        # Check for safety keywords
        unsafe_keywords = ["die", "death", "emergency treatment", "surgery"]
        if any(keyword in response_lower for keyword in unsafe_keywords):
            return 0.0
        
        # Check for appropriate medical disclaimer
        if "consult" in response_lower or "doctor" in response_lower or "provider" in response_lower:
            base_score = 0.7
        else:
            base_score = 0.4
        
        # Length check (too short = poor quality)
        if len(response.split()) < 10:
            base_score = max(0.2, base_score - 0.2)
        elif len(response.split()) > 50:
            base_score = min(0.9, base_score + 0.1)
        
        return min(1.0, base_score)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current environment state for debugging and OpenEnv compliance"""
        return {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "history": self.history[-10:],  # Last 10 events only
            "current_patient_id": self.current_patient["id"] if self.current_patient else None,
            "current_task_id": self.current_task_id,
            "current_patient": self.current_patient,
            "episode_active": self.step_count < self.max_steps if self.current_patient else False
        }