"""
Reconstruct (customer_tweet -> support_reply) pairs for a single brand from
the raw Kaggle "Customer Support on Twitter" CSV (twcs.csv).

Schema of twcs.csv (per Kaggle):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_reply_to_status_id

`inbound` is True for customer tweets, False for brand/support tweets.
`in_reply_to_status_id` on a support tweet points at the customer tweet it
answers. We use that (not response_tweet_id) because it is set directly on
the reply, which is the more reliable direction to join on.

Usage:
    python prepare_data.py --input data/raw/twcs.csv --brand AmericanAir \
        --sample_conversations 5000 --seed 42 --output data/processed/conversations.parquet
"""
import argparse
import re
import pandas as pd


URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = URL_RE.sub("", t)
    t = HANDLE_RE.sub("", t)  # drop @mentions (both customer handles and brand)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    df = df.copy()
    df["author_id"] = df["author_id"].astype(str)

    reply_col = "in_response_to_tweet_id" if "in_response_to_tweet_id" in df.columns else "in_reply_to_status_id"
    support = df[df["author_id"] == brand].copy()
    support = support[support[reply_col].notna()]
    support[reply_col] = support[reply_col].astype(float).astype("Int64")

    customer = df[df["inbound"] == True].copy()  # noqa: E712
    customer["tweet_id"] = customer["tweet_id"].astype("Int64")

    merged = support.merge(
        customer[["tweet_id", "text", "created_at", "author_id"]],
        left_on=reply_col,
        right_on="tweet_id",
        suffixes=("_support", "_customer"),
    )

    # Filter out cases where the "customer" side is actually another brand
    # account replying to itself, or where either side is empty after cleaning.
    merged["customer_text_clean"] = merged["text_customer"].map(clean_text)
    merged["support_text_clean"] = merged["text_support"].map(clean_text)
    merged = merged[
        (merged["customer_text_clean"].str.len() > 3)
        & (merged["support_text_clean"].str.len() > 3)
    ]

    out = merged.rename(
        columns={
            "tweet_id_support": "support_tweet_id",
            "tweet_id_customer": "customer_tweet_id",
            "created_at_support": "support_created_at",
            "created_at_customer": "customer_created_at",
        }
    )[
        [
            "customer_tweet_id",
            "support_tweet_id",
            "customer_text_clean",
            "support_text_clean",
            "customer_created_at",
            "support_created_at",
        ]
    ]
    out = out.drop_duplicates(subset=["customer_tweet_id", "support_tweet_id"])
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--brand", required=True, help="e.g. AmericanAir")
    ap.add_argument("--sample_conversations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    # inbound comes in as string "True"/"False" when dtype=str is forced
    df["inbound"] = df["inbound"].map({"True": True, "False": False})

    pairs = build_pairs(df, args.brand)
    print(f"Reconstructed {len(pairs)} {args.brand} customer->support pairs.")

    if len(pairs) > args.sample_conversations:
        pairs = pairs.sample(n=args.sample_conversations, random_state=args.seed).reset_index(drop=True)
    print(f"Sampled down to {len(pairs)} pairs (seed={args.seed}).")

    pairs.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
