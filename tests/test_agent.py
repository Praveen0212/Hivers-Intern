"""
Unit tests for end-to-end SupportAgent pipeline.
"""

import pytest
from src.agent import SupportAgent


def test_agent_structured_output():
    agent = SupportAgent()
    query = "Where is my package? The tracking number says delivered but nothing is here."
    result = agent.handle_message(query)

    # Check required structured schema
    assert "intent" in result
    assert "confidence" in result
    assert "reply" in result
    assert "decision" in result
    assert "reason" in result
    assert "evidence" in result

    # Validate types and values
    assert isinstance(result["intent"], str)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert len(result["reply"]) > 15
    assert len(result["reason"]) > 10
    assert isinstance(result["evidence"], list)
    if result["evidence"]:
        assert "case_id" in result["evidence"][0]
        assert "similarity" in result["evidence"][0]


def test_agent_escalates_security_query():
    agent = SupportAgent()
    query = "Someone hacked my account and changed the password. Help!"
    result = agent.handle_message(query)
    assert result["decision"] == "ESCALATE"
    assert "account" in result["intent"].lower() or "security" in result["reason"].lower()
