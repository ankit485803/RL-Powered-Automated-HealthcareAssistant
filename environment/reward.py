

"""
Reward Function 
Responsible: Ritu Ranjan
"""

from typing import Dict, Any


def calculate_reward(
    action: Dict[str, Any],
    ground_truth: Dict[str, Any],
    response_quality: float = 0.0,
    response_time_ms: float = 0.0
) -> float:
    """
    Calculate hierarchical reward for healthcare triage action
    
    Reward Structure:
    - Correct ESCALATE: +10
    - Missed emergency (ANSWER when should ESCALATE): -20
    - False alarm (ESCALATE when should not): -5
    - Correct ANSWER with quality: +1 to +5
    - Time penalty: -0.1 per second
    
    Args:
        action: {"decision": str, "response": str}
        ground_truth: {"requires_escalation": bool, "expected_action": str}
        response_quality: Score from LLM-as-judge (0-5)
        response_time_ms: Time taken to respond
    
    Returns:
        Total reward (float)
    """
    
    decision = action.get("decision", "").upper()
    requires_escalation = ground_truth.get("requires_escalation", False)
    
    # Case 1: Agent ESCALATES
    if decision == "ESCALATE":
        if requires_escalation:
            # Correct escalation
            reward = 10.0
        else:
            # False alarm
            reward = -5.0
    
    # Case 2: Agent ANSWERS
    elif decision == "ANSWER":
        if requires_escalation:
            # Missed emergency - severe penalty
            reward = -20.0
        else:
            # Correct answer, reward based on quality
            # Quality score expected from LLM-as-judge (0-5)
            reward = response_quality  # Will be between 0 and 5
    
    else:
        # Invalid decision
        reward = -10.0
    
    # Apply time penalty (0.1 per second, max -2)
    time_penalty = min(response_time_ms / 1000 * 0.1, 2.0)
    reward -= time_penalty
    
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
    # TODO: Ritu - Implement with Gemini LLM-as-judge
    # Consider: accuracy, safety, completeness, clarity
    
    # Placeholder
    return 3.0