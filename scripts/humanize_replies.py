"""
Rewrites the default support replies in golden_unlabeled_v2.csv into natural,
warm, simple English -- the way a real human support agent would type it,
not a corporate press release.

Restores from backup first, then applies safe rewrites only.

Usage:
    python scripts/humanize_replies.py
"""

import re
import os
import shutil
import pandas as pd

INPUT = "golden/golden_unlabeled_v2.csv"
BACKUP = "golden/golden_unlabeled_v2_original.csv"


def clean_reply(text: str) -> str:
    t = str(text).strip()
    if not t or t == "nan":
        return t

    # ── Safe word/phrase substitutions only (not sentence-level strips) ──────
    REPLACEMENTS = [
        # Soften corporate "We're very sorry"
        (r"We're very sorry for", "Sorry about"),
        (r"We are very sorry for", "Sorry about"),
        (r"We're sorry for", "Sorry about"),
        (r"We are sorry for", "Sorry about"),
        # Warm openers
        (r"We're glad", "Great to hear"),
        (r"We are glad", "Great to hear"),
        # Simpler phrasing
        (r"We're showing", "Looks like"),
        (r"We are showing", "Looks like"),
        (r"We're not showing", "Looks like we don't see"),
        (r"We are not showing", "Looks like we don't see"),
        (r"We would be happy to", "Happy to"),
        (r"We'd be happy to", "Happy to"),
        (r"We will have", "We'll get"),
        (r"in order to", "to"),
        (r"at this point in time", "right now"),
        (r"at this time", "right now"),
        (r"Please do not hesitate to", "Feel free to"),
        (r"Please be advised that", "Just so you know,"),
        (r"We are unable to", "We can't"),
        (r"We are able to", "We can"),
        (r"Please follow and DM", "Please DM us"),
        (r"Please continue working with", "Please check with"),
        # Strip trailing filler phrases (only at end of string)
        (r"\.\s*We appreciate your patience\.?$", "."),
        (r"\.\s*Thank you for your patience[^.]*\.?$", "."),
        (r"\.\s*Thank you for reaching out\.?$", "."),
        (r",\s*[A-Z][a-z]{2,12}\.$", "."),    # trailing ", Manny." style names
        (r",\s*[A-Z][a-z]{2,12}!$", "!"),     # trailing ", Manny!" style names
        (r"^(Hi|Hello|Hey),?\s*[A-Z][a-z]{2,12}[,!]\s*", ""),  # "Hi Kevin, " opener
    ]

    for old, new in REPLACEMENTS:
        t = re.sub(old, new, t, flags=re.IGNORECASE)

    # Clean up double spaces and leading punctuation
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"^[,;]\s*", "", t).strip()

    # Capitalize first letter
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    # Ensure terminal punctuation
    if t and t[-1] not in ".!?":
        t += "."

    return t


def main():
    if not os.path.exists(INPUT):
        raise SystemExit(f"Input file not found: {INPUT}")

    # Always restore from backup first so the script is idempotent
    if os.path.exists(BACKUP):
        shutil.copy(BACKUP, INPUT)
        print(f"Restored original from backup: {BACKUP}")
    else:
        shutil.copy(INPUT, BACKUP)
        print(f"Backup saved to {BACKUP}")

    df = pd.read_csv(INPUT)
    original_replies = df["support_text_clean"].copy()
    df["support_text_clean"] = df["support_text_clean"].apply(clean_reply)

    # Show examples
    print("\nSample rewrites:")
    print("-" * 70)
    shown = 0
    for i in range(len(df)):
        orig = str(original_replies.iloc[i])
        new = str(df["support_text_clean"].iloc[i])
        if orig != new and shown < 8:
            print(f"BEFORE: {orig}")
            print(f"AFTER : {new}")
            print()
            shown += 1

    df.to_csv(INPUT, index=False, encoding="utf-8")
    changed = (df["support_text_clean"] != original_replies).sum()
    print(f"Rewritten {changed}/{len(df)} replies. Saved to {INPUT}")
    print("Now run: python scripts/interactive_label.py")


if __name__ == "__main__":
    main()
