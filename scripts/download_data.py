#!/usr/bin/env python3
"""
Download & preparation script for Customer Support on Twitter dataset.
Dataset source: thoughtvector/customer-support-on-twitter (Kaggle)
Mirrored on Hugging Face: SunidhiSriram/twcs

This script safely downloads a reproducible sample (or the full dataset)
without requiring Kaggle API credentials, while also supporting local CSV files.
"""

import argparse
import os
import sys
import requests

DEFAULT_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
DEFAULT_OUTPUT_SAMPLE = os.path.join(RAW_DIR, "twcs_sample.csv")
DEFAULT_SAMPLE_BYTES = 50 * 1024 * 1024  # 50 MB ~ 280,000 tweets (~20,000 AmazonHelp pairs)


def download_stream_sample(url: str, output_path: str, max_bytes: int = DEFAULT_SAMPLE_BYTES) -> str:
    """Download a byte-bounded sample of the CSV stream, truncated at the last newline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[*] Downloading streaming sample ({max_bytes / (1024 * 1024):.1f} MB) from:\n    {url}")
    headers = {"Range": f"bytes=0-{max_bytes}"}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    content = response.content
    last_nl = content.rfind(b"\n")
    if last_nl != -1:
        content = content[:last_nl + 1]

    with open(output_path, "wb") as f:
        f.write(content)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Successfully saved reproducible sample to {output_path} ({size_mb:.2f} MB)")
    return output_path


def download_full_dataset(url: str, output_path: str) -> str:
    """Download full dataset with chunked progress."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[*] Downloading full dataset from:\n    {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_length = r.headers.get("content-length")
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_length:
                        pct = (downloaded / int(total_length)) * 100
                        print(f"\r[Progress] {downloaded / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r[Progress] {downloaded / (1024*1024):.1f} MB", end="", flush=True)
    print("\n[+] Full dataset download complete.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download Customer Support on Twitter dataset.")
    parser.add_argument("--local-path", type=str, default=None, help="Path to existing local twcs.csv file")
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_SAMPLE, help="Destination file path")
    parser.add_argument("--sample-mb", type=int, default=50, help="Megabytes to stream for reproducible sample (default: 50MB)")
    parser.add_argument("--full", action="store_true", help="Download complete ~516MB dataset")
    args = parser.parse_args()

    if args.local_path:
        if not os.path.exists(args.local_path):
            print(f"[-] Error: Specified local file not found: {args.local_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[+] Using local dataset from: {args.local_path}")
        return

    # Check if target already exists
    if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 1024 * 1024:
        print(f"[+] Dataset sample already exists at: {args.output_path} ({os.path.getsize(args.output_path)/(1024*1024):.2f} MB)")
        return

    try:
        if args.full:
            download_full_dataset(DEFAULT_URL, args.output_path)
        else:
            download_stream_sample(DEFAULT_URL, args.output_path, max_bytes=args.sample_mb * 1024 * 1024)
    except Exception as e:
        print(f"[-] Download failed: {e}", file=sys.stderr)
        print("\nManual Alternative Instructions:")
        print("1. Download 'twcs.csv' from Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print(f"2. Place the file at: {DEFAULT_OUTPUT_SAMPLE} (or data/raw/twcs.csv)")
        print("3. Re-run preprocessing with python scripts/prepare_data.py --input-path <path>")
        sys.exit(1)


if __name__ == "__main__":
    main()
