"""
Unit tests for intent classifiers (baselines & main model).
"""

import pytest
import numpy as np
from src.classifier import MajorityBaselineClassifier, TfidfBaselineClassifier, MainIntentClassifier, evaluate_classifier_metrics


@pytest.fixture
def sample_data():
    X = [
        "Where is my package? Tracking is delayed.",
        "Package arrived completely crushed and broken.",
        "I want to return this shirt and get a refund.",
        "Cancel my order immediately before it ships.",
        "My account has been locked and 2FA is failing.",
        "Why was I billed $139 for Prime membership renewal?",
        "Duplicate charge on my credit card statement.",
        "Your customer service is terrible, I want a supervisor."
    ]
    y = [
        "delivery_status",
        "damaged_or_missing",
        "return_and_refund",
        "order_cancellation_or_change",
        "account_and_security",
        "subscription_and_prime",
        "payment_and_billing",
        "general_feedback_or_complaint"
    ]
    return X, y


def test_majority_baseline(sample_data):
    X, y = sample_data
    clf = MajorityBaselineClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert all(isinstance(p, str) for p in preds)
    assert len(set(preds)) == 1  # Unconditional single majority class


def test_tfidf_baseline(sample_data):
    X, y = sample_data
    clf = TfidfBaselineClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(set(y)))


def test_main_intent_classifier(sample_data, tmp_path):
    X, y = sample_data
    clf = MainIntentClassifier(random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)

    # Test single query prediction
    label, conf, dist = clf.predict_single("Where is my package tracking?")
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0
    assert isinstance(dist, dict)

    # Test serialization
    model_file = tmp_path / "model.joblib"
    clf.save(str(model_file))
    loaded = MainIntentClassifier.load(str(model_file))
    loaded_label, loaded_conf, _ = loaded.predict_single("Where is my package tracking?")
    assert loaded_label == label
    assert abs(loaded_conf - conf) < 1e-5


def test_evaluate_classifier_metrics():
    y_true = ["delivery_status", "delivery_status", "return_and_refund"]
    y_pred = ["delivery_status", "return_and_refund", "return_and_refund"]
    metrics = evaluate_classifier_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
