"""
Transparent escalation policy engine.
Evaluates multi-signal criteria:
- Confidence score thresholding
- Retrieval similarity thresholding
- Policy-mandated sensitive intents (security, billing dispute, supervisor complaints)
- High-risk / legal / harm keywords
- Intent-evidence conflict detection
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


DEFAULT_SENSITIVE_INTENTS = {
    "account_and_security",
    "payment_and_billing",
    "general_feedback_or_complaint"
}

HIGH_RISK_PATTERNS = [
    r"\b(lawyer|attorney|sue|lawsuit|legal action|court|small claims)\b",
    r"\b(fraud|scam|stolen card|identity theft|hacked)\b",
    r"\b(police|trespass|harass|assault|profanity|threat)\b",
    r"\b(supervisor|manager|speak to human|real person|call me)\b"
]


class EscalationPolicy:
    """Multi-signal rule and threshold-based escalation arbiter."""

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        similarity_threshold: float = 0.20,
        sensitive_intents: Optional[set] = None,
        require_evidence_consistency: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.sensitive_intents = sensitive_intents or DEFAULT_SENSITIVE_INTENTS
        self.require_evidence_consistency = require_evidence_consistency

    def evaluate(
        self,
        message: str,
        predicted_intent: str,
        confidence: float,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Arbiter deciding AUTO_HANDLE vs ESCALATE with transparent explanatory reasoning.
        Returns: (decision, reason_string, signals_dict)
        """
        reasons = []
        signals = {
            "low_confidence": False,
            "low_similarity": False,
            "sensitive_intent": False,
            "high_risk_keyword": False,
            "intent_mismatch": False,
            "top_similarity": 0.0,
            "confidence": round(confidence, 4)
        }

        # 1. Check sensitive / high-liability intents
        if predicted_intent in self.sensitive_intents:
            signals["sensitive_intent"] = True
            reasons.append(f"Intent '{predicted_intent}' involves sensitive security, financial, or executive complaint policies requiring human agent oversight.")

        # 2. Check high-risk trigger keywords
        msg_lower = message.lower()
        for pattern in HIGH_RISK_PATTERNS:
            match = re.search(pattern, msg_lower)
            if match:
                signals["high_risk_keyword"] = True
                reasons.append(f"Trigger phrase detected ('{match.group(0)}') indicating high risk, legal concern, or supervisor escalation request.")
                break

        # 3. Check model confidence
        if confidence < self.confidence_threshold:
            signals["low_confidence"] = True
            reasons.append(f"Model intent confidence ({confidence:.2f}) was below safe operational threshold ({self.confidence_threshold:.2f}).")

        # 4. Check historical evidence retrieval similarity
        top_sim = max((e.get("similarity", 0.0) for e in retrieved_evidence), default=0.0)
        signals["top_similarity"] = round(top_sim, 4)
        if top_sim < self.similarity_threshold:
            signals["low_similarity"] = True
            reasons.append(f"Highest historical evidence similarity ({top_sim:.2f}) was below grounding threshold ({self.similarity_threshold:.2f}).")

        # 5. Check consistency between predicted intent and top retrieved case
        if self.require_evidence_consistency and retrieved_evidence:
            top_evidence_intent = str(retrieved_evidence[0].get("intent", "unknown")).lower()
            if top_evidence_intent not in ("unknown", "nan", "none", "", predicted_intent.lower()) and top_sim > 0.40:
                signals["intent_mismatch"] = True
                reasons.append(f"Classifier predicted '{predicted_intent}', but top historical match exhibits conflicting intent '{top_evidence_intent}'.")

        # Synthesize final decision
        if reasons:
            decision = "ESCALATE"
            reason = "Escalated: " + " ".join(reasons)
        else:
            decision = "AUTO_HANDLE"
            reason = (
                f"Automated resolution eligible: High confidence ({confidence:.2f} >= {self.confidence_threshold:.2f}), "
                f"well-grounded historical evidence (similarity {top_sim:.2f} >= {self.similarity_threshold:.2f}), "
                f"and routine operational intent ('{predicted_intent}')."
            )

        return decision, reason, signals
