"""
Sample and produce an *unlabeled* golden set from the golden pool only.
Stratified by the intent taxonomy discovered from that same pool so that
rare-but-important intents are represented.

Usage:
    python scripts/sample_golden.py \
        --conversations data/processed/conversations_golden_pool.parquet \
        --taxonomy data/processed/intent_taxonomy_golden.json \
        --n 200 --seed 42 \
        --output golden/golden_unlabeled_v2.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def assign_clusters(texts, taxonomy_path, seed=42):
    """Assign each text to the nearest cluster using TF-IDF (lightweight)."""
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    n_clusters = len(taxonomy)

    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(texts)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    # Map cluster int -> suggested_name
    cluster_to_name = {int(k): v["suggested_name"] for k, v in taxonomy.items()}
    return [cluster_to_name.get(l, f"Cluster_{l}") for l in labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", default="data/processed/conversations_golden_pool.parquet")
    ap.add_argument("--taxonomy", default="data/processed/intent_taxonomy_golden.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="golden/golden_unlabeled_v2.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.conversations)
    print(f"Golden pool has {len(df)} conversations.")

    if len(df) < args.n:
        print(f"Warning: pool has only {len(df)} rows; sampling all of them (< requested {args.n}).")
        args.n = len(df)

    # Assign cluster / suggested intent to each conversation
    texts = df["customer_text_clean"].astype(str).tolist()
    df["suggested_intent"] = assign_clusters(texts, args.taxonomy, seed=args.seed)

    # Stratified sample: proportional to cluster size, at least 1 per cluster
    intents = df["suggested_intent"].unique()
    rng = np.random.default_rng(args.seed)
    sampled_parts = []

    cluster_counts = df["suggested_intent"].value_counts()
    total = cluster_counts.sum()
    remaining = args.n

    for intent in sorted(intents):
        cluster_df = df[df["suggested_intent"] == intent]
        # Proportional allocation, but at least 1
        alloc = max(1, round(len(cluster_df) / total * args.n))
        alloc = min(alloc, len(cluster_df), remaining)
        sampled_idx = rng.choice(len(cluster_df), size=alloc, replace=False)
        sampled_parts.append(cluster_df.iloc[sampled_idx])
        remaining -= alloc
        if remaining <= 0:
            break

    # Fill any remainder from largest cluster
    if remaining > 0:
        pool_remaining = df[~df.index.isin(pd.concat(sampled_parts).index)]
        if len(pool_remaining) > 0:
            extra_idx = rng.choice(len(pool_remaining), size=min(remaining, len(pool_remaining)), replace=False)
            sampled_parts.append(pool_remaining.iloc[extra_idx])

    sample = pd.concat(sampled_parts).reset_index(drop=True)
    sample = sample[["customer_tweet_id", "support_tweet_id",
                      "customer_text_clean", "support_text_clean", "suggested_intent"]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sample.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Sampled {len(sample)} rows -> {args.output}")
    print("\nIntent distribution:")
    print(sample["suggested_intent"].value_counts())


if __name__ == "__main__":
    main()
