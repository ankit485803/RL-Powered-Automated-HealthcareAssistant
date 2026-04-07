
"""
Unit Tests for Reward Function
Responsible: Ritu Ranjan
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.reward import calculate_reward


def test_correct_escalate():
    """Test: Agent correctly escalates emergency"""
    action = {"decision": "ESCALATE", "response": ""}
    ground_truth = {"requires_escalation": True}
    
    reward = calculate_reward(action, ground_truth)
    assert reward == 10.0, f"Expected 10.0, got {reward}"
    print("✓ test_correct_escalate passed")


def test_missed_emergency():
    """Test: Agent answers when should escalate (severe penalty)"""
    action = {"decision": "ANSWER", "response": "Take rest"}
    ground_truth = {"requires_escalation": True}
    
    reward = calculate_reward(action, ground_truth)
    assert reward == -20.0, f"Expected -20.0, got {reward}"
    print("✓ test_missed_emergency passed")


def test_false_alarm():
    """Test: Agent escalates non-emergency"""
    action = {"decision": "ESCALATE", "response": ""}
    ground_truth = {"requires_escalation": False}
    
    reward = calculate_reward(action, ground_truth)
    assert reward == -5.0, f"Expected -5.0, got {reward}"
    print("✓ test_false_alarm passed")


def test_correct_answer_with_quality():
    """Test: Agent correctly answers non-emergency with good quality"""
    action = {"decision": "ANSWER", "response": "Drink water and rest"}
    ground_truth = {"requires_escalation": False}
    
    reward = calculate_reward(action, ground_truth, response_quality=4.0)
    assert reward == 4.0, f"Expected 4.0, got {reward}"
    print("✓ test_correct_answer_with_quality passed")


def test_time_penalty():
    """Test: Time penalty applied correctly"""
    action = {"decision": "ANSWER", "response": "Rest"}
    ground_truth = {"requires_escalation": False}
    
    # 2 second response time = 0.2 penalty
    reward = calculate_reward(action, ground_truth, response_quality=3.0, response_time_ms=2000)
    expected = 3.0 - 0.2  # 2.8
    assert reward == expected, f"Expected {expected}, got {reward}"
    print("✓ test_time_penalty passed")


def run_all_tests():
    print("\n=== Running Reward Function Tests ===\n")
    test_correct_escalate()
    test_missed_emergency()
    test_false_alarm()
    test_correct_answer_with_quality()
    test_time_penalty()
    print("\n=== All Tests Passed ===\n")


if __name__ == "__main__":
    run_all_tests()