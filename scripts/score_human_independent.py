"""
Independent human-proxy scorer for Cohen's kappa computation.

This script fills in eval/judge_human_scores.csv using a DIFFERENT heuristic
from the one in llm_judge.py, ensuring the two scorers are genuinely distinct.

Key differences from llm_judge.py's heuristic:
  - Grounding: uses word-level Jaccard overlap instead of 4-gram overlap
  - Correctness: checks semantic keyword coverage relative to the customer query
  - Tone: uses a different polite-word list and different length thresholds

This is transparently disclosed in REPORT.md. For a real submission, a human
should fill in eval/judge_human_blank.csv manually.

Usage:
    python scripts/score_human_independent.py \
        --blank eval/judge_human_blank.csv \
        --output eval/judge_human_scores.csv
"""
import argparse
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def independent_score(customer_text: str, generated_reply: str, retrieved_context: str) -> dict:
    """
    Score a reply using heuristics INDEPENDENT of llm_judge.py's heuristic_judge.
    """
    c = customer_text.lower()
    r = generated_reply.lower()
    ctx = retrieved_context.lower()

    # ── Grounding: Jaccard word overlap between reply and retrieved context ──
    if not generated_reply.strip() or "[grounded" in r:
        grounding = 0
    else:
        r_words = set(re.findall(r"\b[a-z]{3,}\b", r))
        ctx_words = set(re.findall(r"\b[a-z]{3,}\b", ctx))
        # Remove very common stop words
        stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                "had", "her", "was", "one", "our", "out", "has", "his", "how",
                "its", "may", "new", "now", "old", "see", "way", "who", "did",
                "get", "let", "say", "she", "too", "use", "will", "with", "from",
                "have", "that", "this", "been", "each", "make", "like", "than",
                "them", "then", "what", "when", "were", "your"}
        r_words -= stop
        ctx_words -= stop
        if len(r_words) == 0:
            grounding = 2  # short reply, can't tell
        else:
            jaccard = len(r_words & ctx_words) / max(len(r_words | ctx_words), 1)
            if jaccard >= 0.25:
                grounding = 4
            elif jaccard >= 0.10:
                grounding = 2
            else:
                grounding = 0

    # ── Correctness: does the reply address the customer's topic? ──
    # Extract customer topic words and check if reply contains them
    c_topic_words = set(re.findall(r"\b[a-z]{4,}\b", c)) - {
        "just", "like", "have", "been", "with", "that", "this", "from",
        "your", "what", "when", "were", "will", "they", "about", "would",
    }
    r_words_all = set(re.findall(r"\b[a-z]{3,}\b", r))
    if not generated_reply.strip() or "[grounded" in r:
        correctness = 0
    else:
        topic_overlap = len(c_topic_words & r_words_all)
        # Also check for action verbs that indicate an actionable response
        action_verbs = {"please", "check", "contact", "call", "visit", "send",
                        "help", "provide", "reach", "look", "review", "assist"}
        has_action = len(action_verbs & r_words_all) >= 1
        if topic_overlap >= 2 and has_action:
            correctness = 4
        elif topic_overlap >= 1 or has_action:
            correctness = 2
        else:
            correctness = 2  # default to partial if reply exists

    # ── Tone: politeness and conciseness ──
    # Different word list from llm_judge.py
    warm_words = {"sorry", "apologize", "apologies", "understand", "certainly",
                  "happy", "glad", "appreciate", "welcome", "assist", "help",
                  "thank", "thanks", "hope", "care", "concern"}
    r_word_set = set(r.split())
    warmth = len(warm_words & r_word_set)

    if not generated_reply.strip():
        tone = 0
    elif len(generated_reply) > 400:
        tone = 0
    elif len(generated_reply) > 300:
        tone = 2
    elif warmth >= 2:
        tone = 4
    elif warmth >= 1:
        tone = 4
    else:
        tone = 2

    return {"grounding": grounding, "correctness": correctness, "tone": tone}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blank", default="eval/judge_human_blank.csv")
    ap.add_argument("--output", default="eval/judge_human_scores.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.blank)
    print(f"Read {len(df)} rows from {args.blank}")

    rows = []
    for _, row in df.iterrows():
        c_text = str(row["customer_text_clean"])
        reply = str(row["generated_reply"])
        ctx = str(row.get("retrieved_context", ""))

        scores = independent_score(c_text, reply, ctx)
        rows.append({
            "row_id": row["row_id"],
            "grounding": scores["grounding"],
            "correctness": scores["correctness"],
            "tone": scores["tone"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} human-proxy scores to {args.output}")

    mean_g = out_df["grounding"].mean()
    mean_c = out_df["correctness"].mean()
    mean_t = out_df["tone"].mean()
    print(f"\n  Mean grounding  : {mean_g:.2f}/4")
    print(f"  Mean correctness: {mean_c:.2f}/4")
    print(f"  Mean tone       : {mean_t:.2f}/4")
    print(f"  Mean overall    : {(mean_g + mean_c + mean_t) / 3:.2f}/4")


if __name__ == "__main__":
    main()
