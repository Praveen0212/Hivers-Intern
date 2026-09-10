#!/usr/bin/env python3
"""
Single-command automated evaluation script for the Hiver SDE Intern assignment.
Usage:
    python evaluate.py
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation import evaluate_all

DEFAULT_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden", "golden_set.csv")
DEFAULT_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "train.csv")
DEFAULT_VAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "val.csv")
DEFAULT_REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__))), "reports"


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI customer-support agent across all benchmarks.")
    parser.add_argument("--golden-path", type=str, default=DEFAULT_GOLDEN, help="Path to golden evaluation set CSV")
    parser.add_argument("--train-path", type=str, default=DEFAULT_TRAIN, help="Path to training set CSV")
    parser.add_argument("--val-path", type=str, default=DEFAULT_VAL, help="Path to validation set CSV")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for generated reports")
    args = parser.parse_args()

    if not os.path.exists(args.golden_path):
        print(f"[-] Golden dataset not found at: {args.golden_path}")
        print("    Running scripts/build_golden_set.py to generate it...")
        from scripts.build_golden_set import build_golden_dataset
        build_golden_dataset(args.golden_path)

    if not os.path.exists(args.train_path):
        print(f"[-] Train dataset not found at: {args.train_path}")
        print("    Please run: python scripts/prepare_data.py first.")
        sys.exit(1)

    print("=================================================================")
    print("      RUNNING END-TO-END AUTOMATED EVALUATION PIPELINE           ")
    print("=================================================================\n")

    evaluate_all(
        golden_csv_path=args.golden_path,
        train_csv_path=args.train_path,
        val_csv_path=args.val_path,
        output_dir=args.reports_dir
    )

    print("\n[+] Evaluation complete! Deliverables generated in 'reports/' directory:")
    print("    - reports/results.json")
    print("    - reports/results.csv")
    print("    - reports/confusion_matrix.png")
    print("    - reports/evaluation_report.md")


if __name__ == "__main__":
    main()
