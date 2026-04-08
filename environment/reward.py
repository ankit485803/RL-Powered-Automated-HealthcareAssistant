

"""
Reward Function 
Responsible: Ritu 

Update reward.py - Return 0.0-1.0 Range
Quick sanity check

| Scenario                | Reward |
| ----------------------- | ------ |
| Correct ESCALATE        | 1.0    |
| Missed emergency        | 0.0    |
| False alarm             | 0.2    |
| Good ANSWER (quality=5) | 1.0    |
| Poor ANSWER (quality=0) | 0.3    |



"""

from typing import Dict, Any


def calculate_reward(
    action: Dict[str, Any],
    ground_truth: Dict[str, Any],
    response_quality: float = 0.0,
    response_time_ms: float = 0.0
) -> float:
    """
    Calculate normalized reward for healthcare triage action
    
    Reward Structure (0.0 - 1.0):
    - Correct ESCALATE: 1.0
    - Missed emergency: 0.0
    - False alarm: 0.2
    - Correct ANSWER: 0.3 to 1.0 (based on quality)
    - Time penalty reduces reward
    
    Args:
        action: {"decision": str, "response": str}
        ground_truth: {"requires_escalation": bool, "expected_action": str}
        response_quality: Score from LLM-as-judge (0-5)
        response_time_ms: Time taken to respond
    
    Returns:
        Total reward (float in range 0.0 to 1.0)
    """
    
    decision = action.get("decision", "").upper()
    requires_escalation = ground_truth.get("requires_escalation", False)
    
    # Case 1: Agent ESCALATES
    if decision == "ESCALATE":
        if requires_escalation:
            # Correct escalation
            reward = 1.0
        else:
            # False alarm
            reward = 0.2
    
    # Case 2: Agent ANSWERS
    elif decision == "ANSWER":
        if requires_escalation:
            # Missed emergency - worst case
            reward = 0.0
        else:
            # Correct answer, reward based on quality (0-5 → 0.3-1.0)
            reward = 0.3 + (response_quality / 5.0) * 0.7
    
    else:
        # Invalid decision
        reward = 0.0
    
    # Apply time penalty (0.05 per second, max 0.3)
    time_penalty = min(response_time_ms / 1000 * 0.05, 0.3)
    reward = max(0.0, reward - time_penalty)
    
    return round(reward, 2)


def evaluate_response_quality(
    agent_response: str,
    ground_truth_response: str,
    medical_guidelines: str
) -> float:
    """
    Evaluate quality of LLM response
    To be implemented with LLM-as-judge (Gemini)
    
    Args:
        agent_response: What the agent said
        ground_truth_response: Expected safe response
        medical_guidelines: Reference guidelines
    
    Returns:
        Quality score (0-5)
    """
    # TODO: Ritu 
    # Consider: accuracy, safety, completeness, clarity
    
    # Placeholder
    return 3.0