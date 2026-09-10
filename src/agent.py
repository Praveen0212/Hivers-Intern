"""
End-to-End AI Customer Support Agent for AmazonHelp.
Integrates Intent Classification, Historical Retrieval, Grounded Reply Generation,
and Transparent Escalation Triage into a clean structured response.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import clean_tweet_text
from src.intents import IntentTaxonomy
from src.classifier import MainIntentClassifier
from src.retrieval import HistoricalCaseRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationPolicy

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "classifier.joblib")
DEFAULT_RETRIEVER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "retriever.joblib")
TRAIN_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "train.csv")


class SupportAgent:
    """End-to-end support triage and resolution agent."""

    def __init__(
        self,
        classifier: Optional[MainIntentClassifier] = None,
        retriever: Optional[HistoricalCaseRetriever] = None,
        reply_generator: Optional[GroundedReplyGenerator] = None,
        escalation_policy: Optional[EscalationPolicy] = None
    ):
        self.taxonomy = IntentTaxonomy()
        self.classifier = classifier
        self.retriever = retriever
        self.reply_generator = reply_generator or GroundedReplyGenerator(self.taxonomy)
        self.escalation_policy = escalation_policy or EscalationPolicy(confidence_threshold=0.35, similarity_threshold=0.15)

        # Load or train automatically if components are not supplied
        if self.classifier is None or self.retriever is None:
            self._ensure_models_ready()

    def _ensure_models_ready(self):
        """Loads cached models or fits them from training data."""
        if os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_RETRIEVER_PATH):
            print("[*] Loading cached classifier and retriever models...", file=sys.stderr)
            self.classifier = MainIntentClassifier.load(DEFAULT_MODEL_PATH)
            self.retriever = HistoricalCaseRetriever.load(DEFAULT_RETRIEVER_PATH)
        else:
            print("[*] Cached models not found. Fitting from training dataset...", file=sys.stderr)
            if not os.path.exists(TRAIN_DATA_PATH):
                raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}. Run scripts/prepare_data.py first.")
            train_df = pd.read_csv(TRAIN_DATA_PATH)

            # Fit classifier using training data with matched intents
            # Filter rows with known matched_intent or bootstrap using taxonomy
            labeled_train = train_df[train_df["matched_intent"].notna()].copy()
            if len(labeled_train) < 500:
                # If matched_intent column empty, assign heuristics
                train_df["matched_intent"] = train_df["customer_message"].apply(self.taxonomy.match_rule_based_intent)
                labeled_train = train_df[train_df["matched_intent"].notna()].copy()

            self.classifier = MainIntentClassifier(random_state=42)
            self.classifier.fit(labeled_train["customer_message"].tolist(), labeled_train["matched_intent"].tolist())
            self.classifier.save(DEFAULT_MODEL_PATH)

            # Fit retriever on full historical training cases
            self.retriever = HistoricalCaseRetriever(top_k=3)
            self.retriever.fit(train_df)
            self.retriever.save(DEFAULT_RETRIEVER_PATH)
            print("[+] Models successfully trained and cached.", file=sys.stderr)

    def handle_message(self, message: str) -> Dict[str, Any]:
        """
        Processes an incoming customer message through classification, retrieval,
        reply generation, and escalation triage.
        """
        clean_msg = clean_tweet_text(message)

        # 1. Classify Intent & Confidence
        intent, confidence, dist = self.classifier.predict_single(clean_msg)

        # 2. Retrieve Historical Evidence
        evidence = self.retriever.retrieve(clean_msg, top_k=3)

        # 3. Escalation Triage Decision
        decision, reason, signals = self.escalation_policy.evaluate(
            message=clean_msg,
            predicted_intent=intent,
            confidence=confidence,
            retrieved_evidence=evidence
        )

        # 4. Generate Grounded Reply
        reply_dict = self.reply_generator.generate_reply(
            customer_message=clean_msg,
            intent=intent,
            evidence=evidence,
            decision=decision,
            escalation_reason=reason
        )

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "reply": reply_dict["reply"],
            "decision": decision,
            "reason": reason,
            "evidence": [
                {
                    "case_id": e["case_id"],
                    "similarity": e["similarity"],
                    "snippet": e["customer_message"][:80] + ("..." if len(e["customer_message"]) > 80 else "")
                }
                for e in evidence
            ]
        }


def main():
    parser = argparse.ArgumentParser(description="Run AI Customer Support Agent.")
    parser.add_argument("--message", type=str, required=True, help="Customer inquiry message")
    parser.add_argument("--format", type=str, default="pretty", choices=["pretty", "json"], help="Output display format")
    args = parser.parse_args()

    agent = SupportAgent()
    result = agent.handle_message(args.message)

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("\n=================== AGENT TRIAGE RESPONSE ===================")
        print(f"Intent:     {result['intent']} (Confidence: {result['confidence']:.2f})")
        print(f"Decision:   {result['decision']}")
        print(f"Reason:     {result['reason']}")
        print(f"\nReply:\n{result['reply']}")
        print("\nEvidence Used:")
        for ev in result["evidence"]:
            print(f"  - [{ev['case_id']}] (Sim: {ev['similarity']:.2f}) {ev['snippet']}")
        print("=============================================================\n")


if __name__ == "__main__":
    main()
