"""
Intent taxonomy management, configuration loader, and taxonomy helpers.
"""

import os
from typing import Dict, List, Any, Optional
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "intents.yaml"
)


class IntentTaxonomy:
    """Encapsulates intent definitions, keywords, boundaries, and expected support actions."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.data = self._load_config()
        self.brand = self.data.get("brand", "AmazonHelp")
        self.intents: Dict[str, Dict[str, Any]] = self.data.get("intents", {})
        self.intent_keys: List[str] = list(self.intents.keys())

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Intent config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_intent_names(self) -> Dict[str, str]:
        return {k: v.get("name", k) for k, v in self.intents.items()}

    def get_expected_action(self, intent: str) -> str:
        if intent in self.intents:
            return self.intents[intent].get("expected_action", "ESCALATE")
        return "ESCALATE"

    def get_historical_resolution(self, intent: str) -> str:
        if intent in self.intents:
            return self.intents[intent].get("historical_resolution_pattern", "")
        return ""

    def get_keywords(self, intent: str) -> List[str]:
        if intent in self.intents:
            return self.intents[intent].get("keywords", [])
        return []

    def get_examples(self, intent: str) -> List[str]:
        if intent in self.intents:
            return self.intents[intent].get("examples", [])
        return []

    def match_rule_based_intent(self, text: str) -> Optional[str]:
        """Heuristic pattern matching to assist labeling and bootstrapping."""
        text_lower = text.lower()

        scores = {}
        for intent_key, spec in self.intents.items():
            score = 0
            for kw in spec.get("keywords", []):
                if kw in text_lower:
                    score += 2 if len(kw.split()) > 1 else 1
            if score > 0:
                scores[intent_key] = score

        if not scores:
            return None
        # Return intent with highest keyword score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[0][0]
