"""
Data loading, cleaning, thread reconstruction, and preprocessing module.
Handles Customer Support on Twitter (TWCS) dataset specifically for selected brand.
"""

import html
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_tweet_text(text: str, remove_handles: bool = True) -> str:
    """Clean raw tweet text: decode HTML, remove redundant handles and urls for text modeling."""
    if not isinstance(text, str):
        return ""
    # Decode HTML entities like &amp;, &lt;, &gt;
    text = html.unescape(text)
    # Normalize unicode whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    if remove_handles:
        # Remove mentions like @AmazonHelp, @12345
        text = re.sub(r"@\w+", "", text)
    # Normalize external URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_english_text(text: str) -> bool:
    """Heuristic check ensuring message is predominantly English and substantial."""
    if not text or len(text.strip()) < 10:
        return False
    # Check ASCII / Latin ratio to filter out non-Latin scripts (e.g. Japanese, Arabic)
    ascii_chars = sum(c.isascii() for c in text)
    if (ascii_chars / len(text)) < 0.85:
        return False
    # Check for basic English function words
    common_english = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "am", "are", "was", "were",
        "amazon", "order", "delivery", "prime", "package", "help", "refund"
    }
    words = set(re.findall(r"\b[a-z]{2,}\b", text.lower()))
    if not words.intersection(common_english):
        return False
    return True


def reconstruct_brand_conversations(
    raw_df: pd.DataFrame,
    target_brand: str = "AmazonHelp"
) -> pd.DataFrame:
    """
    Reconstruct customer inquiry -> brand reply pairs from TWCS raw data.
    Ensures that customer inbound tweets are matched with the brand's direct reply.
    """
    # Filter brand responses
    brand_tweets = raw_df[raw_df["author_id"] == target_brand].copy()
    if brand_tweets.empty:
        raise ValueError(f"No records found for target brand '{target_brand}' in dataset.")

    # Create fast index for all tweets by tweet_id
    tweet_map = raw_df.set_index("tweet_id")

    records = []
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row["in_response_to_tweet_id"]
        if pd.isna(parent_id):
            continue
        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue

        if parent_id in tweet_map.index:
            parent = tweet_map.loc[parent_id]
            if isinstance(parent, pd.DataFrame):
                parent = parent.iloc[0]

            # Verify the parent was an inbound customer message
            if bool(parent.get("inbound", False)):
                cust_raw = str(parent.get("text", ""))
                brand_raw = str(brand_row.get("text", ""))

                cust_clean = clean_tweet_text(cust_raw)
                brand_clean = clean_tweet_text(brand_raw, remove_handles=False)

                if is_valid_english_text(cust_clean) and len(brand_clean) > 10:
                    records.append({
                        "inbound_tweet_id": parent_id,
                        "brand_tweet_id": brand_row["tweet_id"],
                        "customer_message_raw": cust_raw,
                        "customer_message": cust_clean,
                        "brand_reply_raw": brand_raw,
                        "brand_reply": brand_clean,
                        "created_at": brand_row.get("created_at", ""),
                    })

    pairs_df = pd.DataFrame(records)
    # Deduplicate based on customer message
    pairs_df = pairs_df.drop_duplicates(subset=["customer_message"]).reset_index(drop=True)
    return pairs_df


def prepare_dataset_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split processed conversations into deterministic train, val, and test partitions."""
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
