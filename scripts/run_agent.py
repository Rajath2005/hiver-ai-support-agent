"""
Run the full agent over the golden set: classify intent (nearest labeled
example by embedding, i.e. a simple kNN classifier trained on the golden
labels themselves via leave-one-out — see decision_log.md for why a
from-scratch classifier is skipped at this label count), retrieve grounding
context, generate a reply, and decide escalation.

Usage:
    python run_agent.py --golden golden/golden_labeled.csv \
        --index data/processed/reply_index --output eval/predictions.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from build_index import retrieve
from escalation import decide_escalation

REPLY_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "reply_generation.txt")


def generate_reply(customer_text: str, retrieved_context: str, top_retrieved_reply: str = "") -> str:
    """Calls the configured LLM if LLM_API_KEY is set; otherwise uses grounded top historical reply."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        # Grounded reply from the most similar historical support resolution
        return top_retrieved_reply if top_retrieved_reply else "[Grounded AmericanAir support reply]"

    with open(REPLY_PROMPT_PATH) as f:
        template = f.read()
    prompt = template.format(retrieved_context=retrieved_context, customer_text=customer_text)

    import anthropic

    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    except Exception as e:
        print(f"LLM generation error ({e}), falling back to top retrieved reply.")
        return top_retrieved_reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    
    # Preload model and index embeddings for speed
    print("Loading embedding model and reply index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_embeddings = np.load(os.path.join(args.index, "embeddings.npy"))
    index_replies = pd.read_parquet(os.path.join(args.index, "replies.parquet"))

    # Pre-encode all queries in golden set
    golden_texts = golden["customer_text_clean"].astype(str).tolist()
    golden_embeddings = model.encode(golden_texts, normalize_embeddings=True, show_progress_bar=False)

    labeled_mask = golden["intent_label"].astype(str).str.len() > 0
    golden_labels = golden["intent_label"].values

    rows = []
    print(f"Generating predictions for {len(golden)} examples...")
    for idx, row in golden.iterrows():
        query = str(row["customer_text_clean"])
        q_emb = golden_embeddings[idx]

        # Leave-one-out kNN over golden intent_label
        sims = golden_embeddings @ q_emb
        sims[idx] = -1.0  # exclude self
        top_knn_idx = np.argsort(-sims)[:5]
        top_knn_labels = [golden_labels[k] for k in top_knn_idx if labeled_mask.iloc[k]]
        intent = max(set(top_knn_labels), key=top_knn_labels.count) if top_knn_labels else "Unknown"

        # Retrieve top-3 grounding examples from index
        idx_sims = index_embeddings @ q_emb
        top_idx = np.argsort(-idx_sims)[:3]
        retrieved = index_replies.iloc[top_idx]
        
        context_str = "\n".join(
            f"- Q: {r.customer_text_clean}\n  A: {r.support_text_clean}"
            for r in retrieved.itertuples()
        )
        top_reply = str(retrieved.iloc[0]["support_text_clean"])
        reply = generate_reply(query, context_str, top_reply)
        escalate, reason = decide_escalation(query)

        rows.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text_clean": query,
                "predicted_intent": intent,
                "generated_reply": reply,
                "escalate": escalate,
                "escalate_reason": reason,
                "retrieved_context": context_str,
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

