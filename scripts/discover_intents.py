"""
Data-driven intent discovery: embed customer queries, cluster, surface top
keywords/examples per cluster, and write a taxonomy skeleton for a human to
name and refine.

This writes `intent_taxonomy.json` with cluster -> {suggested_name, keywords,
example_texts}. A human should review and edit `suggested_name` before it's
used downstream (see decision_log.md #3).

Usage:
    python discover_intents.py --conversations data/processed/conversations.parquet \
        --n_clusters 8 --seed 42 --output data/processed/intent_taxonomy.json
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Rough starting names for an airline brand, purely as a fallback label if a
# human doesn't override; the clustering + keywords is what actually drives
# the taxonomy, not this list.
FALLBACK_NAMES = [
    "FlightStatusOrDelay",
    "Rebooking",
    "RefundOrCancellation",
    "BaggageIssue",
    "Complaint",
    "AccountBookingHelp",
    "ComplimentOrOther",
    "Miscellaneous",
]


def top_keywords(texts, k=8):
    words = Counter()
    for t in texts:
        for w in str(t).lower().split():
            w = w.strip(".,!?\"'")
            if len(w) > 2 and w not in ENGLISH_STOP_WORDS:
                words[w] += 1
    return [w for w, _ in words.most_common(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--n_clusters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.conversations)
    texts = df["customer_text_clean"].tolist()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    except Exception as e:  # pragma: no cover - fallback path
        print(f"sentence-transformers unavailable ({e}); falling back to TF-IDF.")
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=2000, stop_words="english")
        embeddings = vec.fit_transform(texts).toarray()

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(embeddings)

    df["_cluster"] = labels
    taxonomy = {}
    for c in sorted(df["_cluster"].unique()):
        sub = df[df["_cluster"] == c]
        kws = top_keywords(sub["customer_text_clean"], k=8)
        examples = sub["customer_text_clean"].head(5).tolist()
        name = FALLBACK_NAMES[c] if c < len(FALLBACK_NAMES) else f"Cluster_{c}"
        taxonomy[str(c)] = {
            "suggested_name": name,
            "size": int(len(sub)),
            "top_keywords": kws,
            "example_texts": examples,
        }

    with open(args.output, "w") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"Wrote taxonomy skeleton for {args.n_clusters} clusters to {args.output}.")
    print("Review `suggested_name` per cluster against `top_keywords`/`example_texts` before use.")


if __name__ == "__main__":
    main()
