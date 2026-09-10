"""
Historical retrieval engine using TF-IDF cosine similarity over historical customer-support cases.
Retrieves relevant historical queries and their corresponding verified brand resolutions.
"""

import os
import joblib
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalCaseRetriever:
    """Indexes historical conversation pairs and retrieves top-k most similar cases."""

    def __init__(self, top_k: int = 3, max_features: int = 8000):
        self.top_k = top_k
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english"
        )
        self.cases_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None

    def fit(self, cases_df: pd.DataFrame):
        """Fit index on a DataFrame with 'customer_message' and 'brand_reply' columns."""
        self.cases_df = cases_df.copy().reset_index(drop=True)
        if "case_id" not in self.cases_df.columns:
            if "inbound_tweet_id" in self.cases_df.columns:
                self.cases_df["case_id"] = "case_" + self.cases_df["inbound_tweet_id"].astype(str)
            else:
                self.cases_df["case_id"] = [f"case_{i:04d}" for i in range(len(self.cases_df))]

        texts = self.cases_df["customer_message"].fillna("").tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k similar historical cases for an incoming query."""
        if self.tfidf_matrix is None or self.cases_df is None:
            raise ValueError("Retriever index has not been fitted.")

        k = top_k or self.top_k
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            row = self.cases_df.iloc[idx]
            results.append({
                "case_id": str(row.get("case_id", f"case_{idx}")),
                "customer_message": str(row.get("customer_message", "")),
                "brand_resolution": str(row.get("brand_reply", "")),
                "similarity": round(score, 4),
                "intent": str(row.get("matched_intent", row.get("intent", "unknown")))
            })
        return results

    def get_max_similarity(self, results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        return max(r["similarity"] for r in results)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HistoricalCaseRetriever":
        return joblib.load(filepath)
