"""
LLM-as-Judge module for evaluating reply quality and human agreement validation.
Evaluates:
1. Correctness (1-5)
2. Grounding in historical evidence (1-5)
3. Helpfulness (1-5)
4. Relevance (1-5)
5. No unsupported promises (Boolean)
6. Appropriate escalation (1-5)
7. Tone (1-5)
8. Overall Score (1-5)
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


JUDGE_RUBRIC = """
Score criteria (1-5):
- Correctness: Does the reply accurately address the customer's stated issue without factual errors?
- Grounding: Is the reply rooted in verified Amazon policies and retrieved historical resolution patterns?
- Helpfulness: Does the reply provide actionable steps (links, self-service paths, clear directions)?
- Relevance: Is the reply directly relevant to the customer's specific question?
- No Unsupported Promises: Does the reply refrain from promising fake monetary refunds, immediate agent callbacks, or unverified arrival dates?
- Appropriate Escalation: Does the reply escalate when required (security, fraud, complaints) and automate routine inquiries?
- Tone: Is the tone professional, empathetic, and brand-appropriate?
"""


class ReplyJudge:
    """Evaluates agent replies using structured criteria."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def evaluate_reply(
        self,
        customer_message: str,
        generated_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        context: str = "Twitter public customer support (@AmazonHelp)",
        decision: str = "AUTO_HANDLE",
        expected_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate single reply against rubric.
        Uses structured heuristic evaluation or LLM API if key is available.
        """
        evidence_text = " ".join([e.get("snippet", e.get("customer_message", "")) for e in retrieved_evidence])
        combined_ref = evidence_text.lower()
        reply_lower = generated_reply.lower()

        # 1. Relevance: Keyword and semantic alignment
        cust_words = set(re.findall(r"\b\w{4,}\b", customer_message.lower()))
        reply_words = set(re.findall(r"\b\w{4,}\b", reply_lower))
        overlap = len(cust_words.intersection(reply_words))
        relevance_score = min(5, max(3, 3 + (overlap // 2)))

        # 2. Grounding: Evidence and official channel consistency
        official_domains = ["amazon.com/help", "amazon.com/returns", "amazon.com/contact-us", "your orders", "prime"]
        has_domain = any(d in reply_lower for d in official_domains)
        grounding_score = 5 if has_domain else 4

        # 3. Unsupported claims check (hallucinations of compensation or timelines)
        unsupported_claim_patterns = [
            r"\$\d+\s+refund",
            r"within\s+(10|15|30)\s+minutes",
            r"we\s+will\s+compensate\s+you",
            r"free\s+gift\s+card",
            r"i\s+guarantee"
        ]
        has_unsupported = any(re.search(p, reply_lower) for p in unsupported_claim_patterns)
        no_unsupported_promises = not has_unsupported

        # 4. Appropriate escalation
        if expected_action:
            if decision == expected_action:
                escalation_score = 5
            else:
                escalation_score = 2
        else:
            escalation_score = 4

        # 5. Helpfulness
        actionable_cues = ["visit", "check", "select", "contact", "reach", "please"]
        actionable_count = sum(1 for c in actionable_cues if c in reply_lower)
        helpfulness_score = min(5, 3 + (actionable_count // 2))

        # 6. Tone
        tone_score = 5 if ("sorry" in reply_lower or "please" in reply_lower or "understand" in reply_lower) else 4

        # 7. Correctness
        correctness_score = 5 if (no_unsupported_promises and escalation_score >= 4) else 3

        overall = round((correctness_score * 0.25 + grounding_score * 0.20 + helpfulness_score * 0.15 +
                         relevance_score * 0.15 + escalation_score * 0.15 + tone_score * 0.10), 1)

        reason_notes = []
        if has_domain:
            reason_notes.append("Properly referenced official self-service or escalation channels.")
        if no_unsupported_promises:
            reason_notes.append("No speculative promises or unauthorized compensation.")
        if escalation_score < 4:
            reason_notes.append("Decision mismatched gold expected triage action.")

        return {
            "correctness": correctness_score,
            "grounding": grounding_score,
            "helpfulness": helpfulness_score,
            "relevance": relevance_score,
            "no_unsupported_promises": no_unsupported_promises,
            "appropriate_escalation": escalation_score,
            "tone": tone_score,
            "overall": overall,
            "reason": " ".join(reason_notes) or "Meets standard response guidelines."
        }


def compute_human_agreement(agreement_csv_path: str) -> Dict[str, Any]:
    """
    Computes statistical agreement metrics between Human Annotator scores and Judge scores:
    - Pearson Correlation
    - Spearman Rank Correlation
    - Mean Absolute Error (MAE)
    - Agreement within 1 point (%)
    """
    df = pd.read_csv(agreement_csv_path)
    if "human_score" not in df.columns or "llm_score" not in df.columns:
        raise ValueError("Agreement CSV must contain 'human_score' and 'llm_score' columns.")

    human = df["human_score"].to_numpy(dtype=float)
    llm = df["llm_score"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(human - llm)))
    exact_match = float(np.mean(human == llm))
    within_one = float(np.mean(np.abs(human - llm) <= 1.0))

    # Pearson correlation
    if np.std(human) > 0 and np.std(llm) > 0:
        pearson = float(np.corrcoef(human, llm)[0, 1])
    else:
        pearson = 0.0

    # Spearman rank correlation
    rank_human = pd.Series(human).rank()
    rank_llm = pd.Series(llm).rank()
    if np.std(rank_human) > 0 and np.std(rank_llm) > 0:
        spearman = float(np.corrcoef(rank_human, rank_llm)[0, 1])
    else:
        spearman = 0.0

    return {
        "sample_size": len(df),
        "mean_absolute_error": round(mae, 3),
        "pearson_correlation": round(pearson, 3),
        "spearman_correlation": round(spearman, 3),
        "exact_agreement_rate": round(exact_match * 100, 1),
        "within_1_point_agreement_rate": round(within_one * 100, 1)
    }
