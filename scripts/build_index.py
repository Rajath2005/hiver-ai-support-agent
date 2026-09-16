"""
Build a flat (brute-force cosine) embedding index over historical support
replies, keyed by the customer query that prompted each reply. At this scale
(5k-50k pairs) an approximate index (FAISS-IVF/HNSW) buys nothing but
tuning risk (decision_log.md #11).

Usage:
    python build_index.py --conversations data/processed/conversations.parquet \
        --output data/processed/reply_index
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.conversations)
    texts = df["customer_text_clean"].tolist()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"sentence-transformers required for build_index.py ({e}); "
            "pip install sentence-transformers"
        )

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "embeddings.npy"), embeddings)
    df[["customer_text_clean", "support_text_clean"]].to_parquet(
        os.path.join(args.output, "replies.parquet"), index=False
    )
    print(f"Indexed {len(df)} historical replies to {args.output}/")


def retrieve(query: str, index_dir: str, k: int = 3):
    """Helper importable by run_agent.py."""
    from sentence_transformers import SentenceTransformer

    embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
    replies = pd.read_parquet(os.path.join(index_dir, "replies.parquet"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_emb
    top_idx = np.argsort(-sims)[:k]
    return replies.iloc[top_idx][["customer_text_clean", "support_text_clean"]].assign(
        similarity=sims[top_idx]
    )


if __name__ == "__main__":
    main()
