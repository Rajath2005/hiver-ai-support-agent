"""
Split conversations.parquet into two disjoint pools:
  - index_pool  (90%, 4500 rows): used to build the reply retrieval index
  - golden_pool (10%,  500 rows): used to sample the golden evaluation set

A hard assert guarantees zero ID overlap before writing.

Usage:
    python scripts/split_pools.py \
        --input data/processed/conversations.parquet \
        --output_dir data/processed \
        --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/conversations.parquet")
    ap.add_argument("--output_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--golden_frac", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} conversations from {args.input}")

    rng = np.random.default_rng(args.seed)
    shuffled_idx = rng.permutation(len(df))

    n_golden = max(1, int(len(df) * args.golden_frac))
    n_index = len(df) - n_golden

    golden_idx = shuffled_idx[:n_golden]
    index_idx = shuffled_idx[n_golden:]

    golden_pool = df.iloc[golden_idx].reset_index(drop=True)
    index_pool = df.iloc[index_idx].reset_index(drop=True)

    # ─── Hard leakage assertion ────────────────────────────────────────────────
    golden_ids = set(golden_pool["customer_tweet_id"].astype(int))
    index_ids = set(index_pool["customer_tweet_id"].astype(int))
    overlap = golden_ids & index_ids
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} IDs appear in both pools. "
            "This should never happen — file a bug."
        )
    print("OK Disjoint ID check passed -- zero overlap between pools.")

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "conversations_index_pool.parquet")
    golden_path = os.path.join(args.output_dir, "conversations_golden_pool.parquet")

    index_pool.to_parquet(index_path, index=False)
    golden_pool.to_parquet(golden_path, index=False)

    print(f"Index pool  : {len(index_pool):,} rows  -> {index_path}")
    print(f"Golden pool : {len(golden_pool):,} rows  -> {golden_path}")


if __name__ == "__main__":
    main()
