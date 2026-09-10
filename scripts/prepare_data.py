#!/usr/bin/env python3
"""
Prepares customer-support conversation pairs for AmazonHelp from raw Twitter dataset.
Outputs clean paired conversations and train/val/test splits.
"""

import argparse
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import reconstruct_brand_conversations, prepare_dataset_splits
from src.intents import IntentTaxonomy

DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "twcs_sample.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Twitter customer support conversations.")
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT, help="Path to input raw CSV")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DIR, help="Directory to save processed data")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target brand to filter for")
    parser.add_argument("--sample-size", type=int, default=15000, help="Max conversation pairs to retain for training/eval pool")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"[-] Input file not found: {args.input_path}")
        print("Please run: python scripts/download_data.py first.")
        sys.exit(1)

    print(f"[*] Loading raw data from: {args.input_path}")
    raw_df = pd.read_csv(args.input_path, low_memory=False)
    print(f"[*] Raw dataset loaded: {len(raw_df):,} tweets.")

    print(f"[*] Reconstructing conversation pairs for '{args.brand}'...")
    pairs_df = reconstruct_brand_conversations(raw_df, target_brand=args.brand)
    print(f"[+] Reconstructed {len(pairs_df):,} clean English conversation pairs.")

    if len(pairs_df) > args.sample_size:
        pairs_df = pairs_df.sample(n=args.sample_size, random_state=args.random_seed).reset_index(drop=True)
        print(f"[*] Subsampled to {len(pairs_df):,} pairs for rapid, reproducible execution.")

    # Assign bootstrap pseudo-label if matched, or flag for labeling
    taxonomy = IntentTaxonomy()
    pairs_df["matched_intent"] = pairs_df["customer_message"].apply(taxonomy.match_rule_based_intent)

    os.makedirs(args.output_dir, exist_ok=True)
    all_path = os.path.join(args.output_dir, "amazon_conversations.csv")
    pairs_df.to_csv(all_path, index=False)
    print(f"[+] Saved full processed conversations to: {all_path}")

    # Create train, val, test splits
    train_df, val_df, test_df = prepare_dataset_splits(pairs_df, test_size=0.2, val_size=0.1, random_state=args.random_seed)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Splits generated successfully:")
    print(f"    - Train: {len(train_df):,} rows -> {train_path}")
    print(f"    - Val:   {len(val_df):,} rows -> {val_path}")
    print(f"    - Test:  {len(test_df):,} rows -> {test_path}")


if __name__ == "__main__":
    main()
