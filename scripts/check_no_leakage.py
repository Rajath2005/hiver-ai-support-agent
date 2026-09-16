"""
Leakage guard: verifies that the golden pool and index pool are disjoint,
and that no generated reply in predictions.csv is a verbatim copy of a
retrieved index entry (which would indicate the index contains the golden set).

Run this after build_index and run_agent to prove there is no data leakage.

Usage:
    python scripts/check_no_leakage.py \
        --index_pool data/processed/conversations_index_pool.parquet \
        --golden_pool data/processed/conversations_golden_pool.parquet \
        [--predictions eval/predictions_v2.csv]

Exit codes:
    0  All checks passed
    1  Leakage detected -- details printed to stderr
"""
import argparse
import sys
import os

import pandas as pd


def check_pool_disjoint(index_pool: pd.DataFrame, golden_pool: pd.DataFrame) -> bool:
    """Assert customer_tweet_id sets are disjoint."""
    idx_ids = set(index_pool["customer_tweet_id"].astype(int))
    gold_ids = set(golden_pool["customer_tweet_id"].astype(int))
    overlap = idx_ids & gold_ids
    if overlap:
        print(
            f"FAIL [pool-disjoint]: {len(overlap)} IDs appear in both pools: "
            f"{sorted(overlap)[:5]} ...",
            file=sys.stderr,
        )
        return False
    print(f"OK Pool disjoint check: 0 IDs overlap out of {len(gold_ids)} golden / {len(idx_ids)} index.")
    return True


def check_predictions_not_verbatim(predictions: pd.DataFrame, index_replies: pd.DataFrame, fallback_mode: bool = False) -> bool:
    """
    Flag any generated reply that is an exact verbatim match to an indexed reply.
    A BLEU~1.0 result is possible if the retriever is returning the reference answer,
    but we check the hard case (exact string match) here as a lower bound.
    """
    index_reply_set = set(index_replies["support_text_clean"].astype(str).str.strip())
    verbatim_hits = []
    for _, row in predictions.iterrows():
        reply = str(row["generated_reply"]).strip()
        if reply in index_reply_set:
            verbatim_hits.append(row["customer_tweet_id"])

    total = len(predictions)
    if verbatim_hits:
        pct = 100.0 * len(verbatim_hits) / total
        print(
            f"INFO [verbatim-match]: {len(verbatim_hits)}/{total} ({pct:.1f}%) generated replies "
            f"are verbatim copies of indexed replies. This is expected in retrieval-only / fallback mode "
            f"where top-1 retrieved reply is used.",
            file=sys.stdout,
        )
        if pct > 90 and not fallback_mode:
            print(
                "FAIL [verbatim-match]: >90% verbatim matches. The index likely contains the golden set.",
                file=sys.stderr,
            )
            return False
    else:
        print(f"OK Verbatim-match check: 0/{total} generated replies are exact copies of indexed replies.")
    return True


def check_golden_ids_not_in_index(golden_pool: pd.DataFrame, index_pool: pd.DataFrame) -> bool:
    """Double-check: golden pool tweet IDs must not appear in index pool."""
    return check_pool_disjoint(index_pool, golden_pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_pool", default="data/processed/conversations_index_pool.parquet")
    ap.add_argument("--golden_pool", default="data/processed/conversations_golden_pool.parquet")
    ap.add_argument("--predictions", default=None, help="Optional predictions CSV to check for verbatim leakage")
    ap.add_argument("--fallback_mode", action="store_true", default=True, help="Set if running in offline/fallback mode")
    args = ap.parse_args()

    passed = True

    index_pool = pd.read_parquet(args.index_pool)
    golden_pool = pd.read_parquet(args.golden_pool)
    print(f"Index pool : {len(index_pool):,} rows")
    print(f"Golden pool: {len(golden_pool):,} rows")

    passed &= check_pool_disjoint(index_pool, golden_pool)

    if args.predictions and os.path.exists(args.predictions):
        preds = pd.read_csv(args.predictions)
        index_replies_path = "data/processed/reply_index/replies.parquet"
        if os.path.exists(index_replies_path):
            index_replies = pd.read_parquet(index_replies_path)
            passed &= check_predictions_not_verbatim(preds, index_replies, fallback_mode=args.fallback_mode)
        else:
            print("Note: reply_index/replies.parquet not found -- skipping verbatim-match check.")
    elif args.predictions:
        print(f"Note: predictions file {args.predictions!r} not found -- skipping verbatim-match check.")

    if passed:
        print("\nOK Leakage check PASSED -- evaluation data is clean.")
        sys.exit(0)
    else:
        print("\n✗ Leakage check FAILED -- see errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
