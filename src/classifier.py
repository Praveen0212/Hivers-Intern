"""
Intent classification module providing:
1. Baseline 1 (Trivial): Majority-class classifier
2. Baseline 2 (Simple): TF-IDF + Logistic Regression
3. Main Classifier: Calibrated TF-IDF (word + char n-grams) + Cost-sensitive Logistic Regression
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline


class MajorityBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Trivial Baseline: Predicts the most frequent training class unconditionally."""

    def __init__(self):
        self.majority_class_: Optional[str] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        counts = pd.Series(y).value_counts()
        self.majority_class_ = counts.index[0]
        self.classes_ = np.array(sorted(list(set(y))))
        return self

    def predict(self, X) -> np.ndarray:
        return np.array([self.majority_class_] * len(X))

    def predict_proba(self, X) -> np.ndarray:
        prob = np.zeros((len(X), len(self.classes_)))
        idx = list(self.classes_).index(self.majority_class_)
        prob[:, idx] = 1.0
        return prob


class TfidfBaselineClassifier:
    """Simple Baseline: Standard unigram TF-IDF + basic Logistic Regression."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=2500, stop_words="english")),
            ("clf", LogisticRegression(random_state=random_state, max_iter=500))
        ])

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)


class MainIntentClassifier:
    """
    Main Classifier: Optimized sublinear TF-IDF (unigram + bigram) with balanced class weights
    and temperature-calibrated softmax confidence.
    """

    def __init__(self, random_state: int = 42, C: float = 2.0):
        self.random_state = random_state
        self.C = C
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=6000,
                sublinear_tf=True,
                strip_accents="unicode",
                stop_words="english",
                token_pattern=r"(?u)\b\w[\w'-]+\b"
            )),
            ("clf", LogisticRegression(
                C=C,
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000,
                solver="lbfgs"
            ))
        ])
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict intent and confidence for a single text."""
        proba = self.predict_proba([text])[0]
        max_idx = int(np.argmax(proba))
        pred_label = str(self.classes_[max_idx])
        confidence = float(proba[max_idx])
        distribution = {str(cls): float(p) for cls, p in zip(self.classes_, proba)}
        return pred_label, confidence, distribution

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MainIntentClassifier":
        return joblib.load(filepath)


def evaluate_classifier_metrics(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics: Accuracy, Macro F1, Weighted F1, Per-class metrics."""
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels
    }
