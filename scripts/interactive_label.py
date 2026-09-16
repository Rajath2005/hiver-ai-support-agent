"""
Interactive, resumable CLI for hand-labeling the golden evaluation set.

Each row is shown one at a time. You type the label, press Enter, and the
row is appended to the output CSV immediately (so Ctrl+C at any point is
safe -- all previously labeled rows are saved).

Usage:
    python scripts/interactive_label.py \
        --input golden/golden_unlabeled_v2.csv \
        --output golden/golden_labeled_v2.csv \
        [--intents data/processed/intent_taxonomy_golden.json]

Controls:
    - At each prompt: type your answer and press Enter
    - Press Ctrl+C to save and exit; re-run the same command to resume
    - At the reference_reply prompt: press Enter with no input to accept
      the historical support reply as-is, or type to override it

Columns written to output CSV:
    customer_tweet_id, support_tweet_id, customer_text_clean,
    support_text_clean, suggested_intent,
    intent_label, escalate, escalate_reason, reference_reply
"""
import argparse
import csv
import json
import os
import sys

VALID_ESCALATE = {"y", "n", "yes", "no"}
OUTPUT_COLS = [
    "customer_tweet_id", "support_tweet_id",
    "customer_text_clean", "support_text_clean", "suggested_intent",
    "intent_label", "escalate", "escalate_reason", "reference_reply",
]


def load_done_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["customer_tweet_id"])
    return done


def append_row(output_path: str, row: dict):
    file_exists = os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in OUTPUT_COLS})


def prompt(msg: str, default: str = "") -> str:
    """Prompt the user. Returns stripped input; returns `default` on empty enter."""
    try:
        val = input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt
    return val if val else default


def load_intents(taxonomy_path: str) -> list:
    if not taxonomy_path or not os.path.exists(taxonomy_path):
        return []
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    return sorted(v["suggested_name"] for v in taxonomy.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="golden/golden_unlabeled_v2.csv")
    ap.add_argument("--output", default="golden/golden_labeled_v2.csv")
    ap.add_argument("--intents", default="data/processed/intent_taxonomy_golden.json")
    args = ap.parse_args()

    # Load input rows
    import pandas as pd
    unlabeled = pd.read_csv(args.input)
    done_ids = load_done_ids(args.output)
    todo = unlabeled[~unlabeled["customer_tweet_id"].astype(str).isin(done_ids)]

    known_intents = load_intents(args.intents)
    total = len(unlabeled)
    done_count = len(done_ids)

    if len(todo) == 0:
        print(f"All {total} rows are already labeled in {args.output}. Nothing to do.")
        return

    print("=" * 70)
    print("HIVER GOLDEN SET -- Interactive Labeling CLI")
    print("=" * 70)
    print(f"  Total rows : {total}")
    print(f"  Already done: {done_count}")
    print(f"  Remaining  : {len(todo)}")
    print()
    if known_intents:
        print("  Known intents:")
        for i, name in enumerate(known_intents, 1):
            print(f"    [{i}] {name}")
    print()
    print("  Press Ctrl+C at any time to save and quit. Re-run to resume.")
    print("=" * 70)

    try:
        for position, (_, row) in enumerate(todo.iterrows(), start=done_count + 1):
            print(f"\n{'─' * 70}")
            print(f"  Row {position}/{total}  |  ID: {row['customer_tweet_id']}")
            print(f"{'─' * 70}")
            print(f"\n  CUSTOMER TWEET:\n")
            print(f"    {row['customer_text_clean']}\n")
            print(f"  HISTORICAL SUPPORT REPLY:\n")
            print(f"    {row['support_text_clean']}\n")
            print(f"  Suggested intent: {row.get('suggested_intent', '?')}\n")

            # ── Intent label ────────────────────────────────────────────────
            if known_intents:
                intent_hint = f"  Intent [{'/'.join(known_intents)}]"
            else:
                intent_hint = "  Intent label"

            while True:
                intent = prompt(
                    f"{intent_hint}\n  (Enter number 1-{len(known_intents)} or type name, "
                    f"default={row.get('suggested_intent', '')}): ",
                    default=str(row.get("suggested_intent", "")),
                )
                # Allow number shortcuts
                if intent.isdigit() and known_intents:
                    idx = int(intent) - 1
                    if 0 <= idx < len(known_intents):
                        intent = known_intents[idx]
                if intent:
                    break
                print("  Intent cannot be empty.")

            # ── Escalate ─────────────────────────────────────────────────────
            while True:
                esc_raw = prompt("  Escalate to human? [y/n, default=n]: ", default="n").lower()
                if esc_raw in VALID_ESCALATE:
                    escalate = "yes" if esc_raw in {"y", "yes"} else "no"
                    break
                print("  Please enter y or n.")

            # ── Escalate reason ──────────────────────────────────────────────
            if escalate == "yes":
                reason = prompt(
                    "  Escalation reason (why this needs a human): ",
                    default="Requires human follow-up."
                )
            else:
                reason = prompt(
                    "  Skip reason (why auto-handle is OK, or Enter to skip): ",
                    default=""
                )

            # ── Reference reply ──────────────────────────────────────────────
            print(f"\n  Default reference reply (press Enter to accept):")
            print(f"    \"{row['support_text_clean']}\"")
            ref_reply = prompt(
                "  Reference reply (Enter=accept, or type to override): ",
                default=str(row["support_text_clean"]),
            )

            # ── Write row ────────────────────────────────────────────────────
            labeled_row = {
                "customer_tweet_id": row["customer_tweet_id"],
                "support_tweet_id": row["support_tweet_id"],
                "customer_text_clean": row["customer_text_clean"],
                "support_text_clean": row["support_text_clean"],
                "suggested_intent": row.get("suggested_intent", ""),
                "intent_label": intent,
                "escalate": escalate,
                "escalate_reason": reason,
                "reference_reply": ref_reply,
            }
            append_row(args.output, labeled_row)
            print(f"  OK Saved.")

    except KeyboardInterrupt:
        pass

    # Summary
    done_now = load_done_ids(args.output)
    print(f"\n\n  Labeled {len(done_now)}/{total} rows total. Output: {args.output}")
    print("  Re-run this script to continue labeling where you left off.\n")


if __name__ == "__main__":
    main()
