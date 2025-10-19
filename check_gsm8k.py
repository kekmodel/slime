#!/usr/bin/env python3
"""Check GSM8K dataset structure"""

import pandas as pd
import sys

def check_dataset(path):
    print(f"Checking dataset: {path}\n")

    try:
        df = pd.read_parquet(path)

        print("=" * 60)
        print("DATASET INFO")
        print("=" * 60)
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}\n")

        print("=" * 60)
        print("COLUMN TYPES")
        print("=" * 60)
        print(df.dtypes)
        print()

        print("=" * 60)
        print("FIRST 3 ROWS (Sample)")
        print("=" * 60)
        for idx in range(min(3, len(df))):
            print(f"\n--- Row {idx} ---")
            for col in df.columns:
                value = df.iloc[idx][col]
                if isinstance(value, str) and len(value) > 100:
                    print(f"{col}: {value[:100]}...")
                else:
                    print(f"{col}: {value}")

        print("\n" + "=" * 60)
        print("RECOMMENDED SETTINGS FOR SLIME")
        print("=" * 60)

        # Detect the right keys
        possible_input_keys = ['question', 'messages', 'prompt', 'input']
        possible_label_keys = ['answer', 'label', 'output', 'response']

        input_key = None
        label_key = None

        for key in possible_input_keys:
            if key in df.columns:
                input_key = key
                break

        for key in possible_label_keys:
            if key in df.columns:
                label_key = key
                break

        if input_key:
            print(f"--input-key {input_key}")
        else:
            print("⚠️  Could not auto-detect input key")

        if label_key:
            print(f"--label-key {label_key}")
        else:
            print("⚠️  Could not auto-detect label key")

        # Check if messages format
        if 'messages' in df.columns:
            print("--apply-chat-template  # (messages format detected)")

        print("\n")

    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "gsm8k/train.parquet"

    check_dataset(path)
