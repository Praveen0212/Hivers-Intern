"""
Unit tests for transparent escalation policy.
"""

import pytest
from src.escalation import EscalationPolicy


def test_escalation_on_sensitive_intent():
    policy = EscalationPolicy()
    decision, reason, signals = policy.evaluate(
        message="I cannot log into my account.",
        predicted_intent="account_and_security",
        confidence=0.90,
        retrieved_evidence=[{"case_id": "c1", "similarity": 0.85, "intent": "account_and_security"}]
    )
    assert decision == "ESCALATE"
    assert signals["sensitive_intent"] is True
    assert "sensitive security" in reason


def test_escalation_on_high_risk_keyword():
    policy = EscalationPolicy()
    decision, reason, signals = policy.evaluate(
        message="I will get my lawyer to file a lawsuit against Amazon!",
        predicted_intent="delivery_status",
        confidence=0.88,
        retrieved_evidence=[{"case_id": "c1", "similarity": 0.50, "intent": "delivery_status"}]
    )
    assert decision == "ESCALATE"
    assert signals["high_risk_keyword"] is True
    assert "Trigger phrase" in reason


def test_escalation_on_low_confidence():
    policy = EscalationPolicy(confidence_threshold=0.50)
    decision, reason, signals = policy.evaluate(
        message="Vague request here.",
        predicted_intent="delivery_status",
        confidence=0.25,
        retrieved_evidence=[{"case_id": "c1", "similarity": 0.40, "intent": "delivery_status"}]
    )
    assert decision == "ESCALATE"
    assert signals["low_confidence"] is True
    assert "Model intent confidence" in reason


def test_auto_handle_on_clean_query():
    policy = EscalationPolicy(confidence_threshold=0.35, similarity_threshold=0.15)
    decision, reason, signals = policy.evaluate(
        message="Where is my package? The tracking has not updated since yesterday.",
        predicted_intent="delivery_status",
        confidence=0.60,
        retrieved_evidence=[{"case_id": "c1", "similarity": 0.45, "intent": "delivery_status"}]
    )
    assert decision == "AUTO_HANDLE"
    assert "Automated resolution eligible" in reason
