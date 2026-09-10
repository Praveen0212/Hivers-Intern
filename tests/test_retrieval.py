"""
Unit tests for historical case retrieval module.
"""

import pytest
import pandas as pd
from src.retrieval import HistoricalCaseRetriever


@pytest.fixture
def sample_cases_df():
    return pd.DataFrame([
        {
            "case_id": "case_001",
            "customer_message": "Where is my package? It was supposed to arrive yesterday.",
            "brand_reply": "Please check your tracking details under Your Orders.",
            "matched_intent": "delivery_status"
        },
        {
            "case_id": "case_002",
            "customer_message": "My order arrived with broken glass everywhere.",
            "brand_reply": "Please visit amazon.com/returns for a replacement.",
            "matched_intent": "damaged_or_missing"
        },
        {
            "case_id": "case_003",
            "customer_message": "How do I return this item at Whole Foods?",
            "brand_reply": "You can generate a return QR code on the Returns Center.",
            "matched_intent": "return_and_refund"
        }
    ])


def test_retriever_fit_and_retrieve(sample_cases_df):
    retriever = HistoricalCaseRetriever(top_k=2)
    retriever.fit(sample_cases_df)

    results = retriever.retrieve("Where is my package delayed?")
    assert len(results) == 2
    assert results[0]["case_id"] == "case_001"  # Best semantic match
    assert results[0]["similarity"] > 0.1
    assert "brand_resolution" in results[0]


def test_retriever_serialization(sample_cases_df, tmp_path):
    retriever = HistoricalCaseRetriever(top_k=2)
    retriever.fit(sample_cases_df)

    save_path = tmp_path / "retriever.joblib"
    retriever.save(str(save_path))

    loaded = HistoricalCaseRetriever.load(str(save_path))
    results = loaded.retrieve("Where is my package?")
    assert len(results) == 2
    assert results[0]["case_id"] == "case_001"
