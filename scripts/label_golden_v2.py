"""
Improved automated labeling for golden_unlabeled_v2.csv → golden_labeled_v2.csv.

This is a more careful rule-based labeler than label_golden.py (v1).
Improvements:
  - Multi-word phrase matching for intents (not just single keywords)
  - Priority-ordered intent rules (most specific first)
  - Escalation logic aligned with scripts/escalation.py
  - Considers both customer AND support text for context clues
  - Reference reply defaults to historical support_text_clean

IMPORTANT: This is automated labeling, not human hand-labeling.
This is transparently disclosed in REPORT.md and decision_log.md.
For a genuine evaluation, use scripts/interactive_label.py instead.

Usage:
    python scripts/label_golden_v2.py \
        --input golden/golden_unlabeled_v2.csv \
        --output golden/golden_labeled_v2.csv
"""
import argparse
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ─── Intent classification rules (priority-ordered, most specific first) ──────

INTENT_RULES = [
    # 1. Baggage — specific baggage vocabulary
    {
        "intent": "BaggageIssue",
        "customer_patterns": [
            r"\bbag\b", r"\bbags\b", r"\bbaggage\b", r"\bluggage\b",
            r"\bsuitcase\b", r"\bcarousel\b", r"\bchecked bag\b",
            r"\bcarry.?on\b", r"\blost.{0,10}bag\b", r"\bdamaged.{0,10}bag\b",
            r"\bmissing.{0,10}luggage\b", r"\bbag.{0,10}claim\b",
            r"\bbag.{0,10}fee\b", r"\boverhead bin\b",
        ],
        "support_patterns": [
            r"\bbag\b", r"\bbaggage\b", r"\bluggage\b", r"\bcarousel\b",
        ],
    },
    # 2. Seating & Boarding
    {
        "intent": "SeatingAndBoarding",
        "customer_patterns": [
            r"\bseat\b", r"\bseats\b", r"\bseating\b", r"\bbulkhead\b",
            r"\bupgrade\b", r"\bboarding\b", r"\bboard\b", r"\bfirst class\b",
            r"\bexit row\b", r"\bbusiness class\b", r"\baisle\b", r"\bwindow\b",
            r"\bmiddle seat\b", r"\bleg.?room\b", r"\boverbooked\b",
            r"\bstandby\b", r"\bpriority boarding\b",
        ],
        "support_patterns": [
            r"\bseat\b", r"\bupgrade\b", r"\bboarding\b", r"\bstandby\b",
        ],
    },
    # 3. Flight Delays & Disruptions
    {
        "intent": "FlightDelaysAndDisruptions",
        "customer_patterns": [
            r"\bdelay\b", r"\bdelayed\b", r"\bdelays\b", r"\blate\b",
            r"\bcancell?ed\b", r"\bcancellation\b", r"\bmissed connection\b",
            r"\bdiverted\b", r"\btarmac\b", r"\bstuck\b", r"\bstranded\b",
            r"\bmechanical\b", r"\bweather\b", r"\bhours? late\b",
            r"\breboo[k]\b", r"\brescheduled?\b", r"\bmissed.{0,10}flight\b",
            r"\bwaiting\b.{0,20}\b(hour|gate|airport)\b",
            r"\bon time\b",
        ],
        "support_patterns": [
            r"\bdelay\b", r"\breboo[k]\b", r"\bcancel\b", r"\bdivert\b",
            r"\bdisruption\b",
        ],
    },
    # 4. Ticketing & Refunds
    {
        "intent": "TicketingAndRefunds",
        "customer_patterns": [
            r"\brefund\b", r"\bvoucher\b", r"\bticket\b", r"\bfare\b",
            r"\bprice\b", r"\bcost\b", r"\bcharge\b", r"\b\$\d+\b",
            r"\bbook\b", r"\bbooking\b", r"\breservation\b",
            r"\bchange fee\b", r"\breceipt\b", r"\bmiles\b",
            r"\baadvantage\b", r"\bloyalty\b", r"\bpoints\b",
            r"\bcompensation\b", r"\breimburse\b", r"\bcredit\b",
            r"\bbasic economy\b", r"\bpurchase\b",
        ],
        "support_patterns": [
            r"\brefund\b", r"\bvoucher\b", r"\bfare\b", r"\bticket\b",
            r"\breservation\b", r"\b800-433-7300\b",
        ],
    },
    # 5. Direct Assistance / DM request
    {
        "intent": "DirectAssistanceOrDM",
        "customer_patterns": [
            r"\bphone number\b", r"\bcall center\b", r"\b[dD][mM]\b",
            r"\bdirect message\b", r"\bsent dm\b", r"\brecord locator\b",
            r"\bbooking reference\b", r"\bcontact number\b",
            r"\bspeak to\b", r"\bcall\b.{0,10}\bnumber\b",
        ],
        "support_patterns": [
            r"\b[dD][mM]\b", r"\bdirect message\b", r"\brecord locator\b",
        ],
    },
    # 6. Customer Service Complaint (explicit poor-service complaints)
    {
        "intent": "CustomerServiceComplaint",
        "customer_patterns": [
            r"\brude\b", r"\battitude\b", r"\bterrible service\b",
            r"\bworst.{0,15}(service|airline|customer)\b",
            r"\bshame on you\b", r"\bdisrespectful\b", r"\bunhelpful\b",
            r"\bhorrible\b.{0,10}service\b", r"\bdisgusting\b",
            r"\bunacceptable\b", r"\bawful\b.{0,10}service\b",
            r"\bnever fly\b", r"\bnever again\b",
            r"\bpissing me off\b", r"\bfuck\b", r"\bshit\b",
        ],
        "support_patterns": [],
    },
    # 7. Compliment or Appreciation
    {
        "intent": "ComplimentOrAppreciation",
        "customer_patterns": [
            r"\bthank you\b", r"\bthanks\b", r"\bgreat job\b",
            r"\bkudos\b", r"\bwonderful\b", r"\bamazing\b",
            r"\blove flying\b", r"\bshout.?out\b", r"\bbest crew\b",
            r"\bphenomenal\b", r"\bexcellent\b", r"\bappreciate\b",
            r"\bcredit where\b", r"\bcan'?t wait\b",
            r"\bhappy\b.{0,15}\b(fly|travel|trip)\b",
        ],
        "support_patterns": [],
    },
]


def classify_intent(c_text: str, s_text: str, suggested: str) -> str:
    """Classify intent using priority-ordered rule matching."""
    c_lower = c_text.lower()
    s_lower = s_text.lower()

    best_intent = None
    best_score = 0

    for rule in INTENT_RULES:
        customer_hits = sum(
            1 for p in rule["customer_patterns"]
            if re.search(p, c_lower, re.IGNORECASE)
        )
        support_hits = sum(
            1 for p in rule.get("support_patterns", [])
            if re.search(p, s_lower, re.IGNORECASE)
        )
        # Customer text is primary signal; support text is secondary confirmation
        score = customer_hits * 2 + support_hits

        if score > best_score:
            best_score = score
            best_intent = rule["intent"]

    if best_intent and best_score >= 2:
        return best_intent

    # Fallback: use suggested intent from clustering, mapped to our taxonomy
    CLUSTER_TO_INTENT = {
        "FlightStatusOrDelay": "FlightDelaysAndDisruptions",
        "Rebooking": "TicketingAndRefunds",
        "RefundOrCancellation": "TicketingAndRefunds",
        "BaggageIssue": "BaggageIssue",
        "Complaint": "CustomerServiceComplaint",
        "AccountBookingHelp": "GeneralFlightFeedback",
        "ComplimentOrOther": "ComplimentOrAppreciation",
        "Miscellaneous": "GeneralFlightFeedback",
    }
    return CLUSTER_TO_INTENT.get(suggested, "GeneralFlightFeedback")


# ─── Escalation rules (aligned with scripts/escalation.py) ───────────────────

SAFETY_TERMS = [
    r"\bunsafe\b", r"\bemergency\b", r"\bmedical\b", r"\bassault\b",
    r"\bthreat(en)?\b", r"\bhurt\b", r"\binjur(y|ed)\b", r"\bsick\b",
    r"\bdanger\b",
]
LEGAL_TERMS = [
    r"\blawyer\b", r"\blawsuit\b", r"\bsue\b", r"\blegal\b",
    r"\battorney\b", r"\bdot complaint\b", r"\bfaa\b",
]
ESCALATION_REQUEST = [
    r"\bsupervisor\b", r"\bmanager\b", r"\bhuman\b",
    r"\breal person\b", r"\bspeak to someone\b", r"\brepresentative\b",
]
STRONG_NEGATIVE = [
    r"\bdisgust(ing)?\b", r"\bterrible\b", r"\bworst\b",
    r"\bnever again\b", r"\bscam\b", r"\bfraud\b", r"\bunacceptable\b",
    r"\bstranded\b", r"\bruined\b", r"\bstolen\b", r"\bnever fly\b",
]
LOST_BAGGAGE = [
    r"\blost.{0,10}bag\b", r"\blost.{0,10}luggage\b",
    r"\bdamaged.{0,10}bag\b", r"\bmissing.{0,10}luggage\b",
]


def _any_match(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_escalation(c_text: str) -> tuple:
    """Determine escalation. Returns (escalate_str, reason)."""
    if _any_match(SAFETY_TERMS, c_text):
        return "yes", "Medical, injury, or safety-related incident requiring immediate human intervention."
    if _any_match(LEGAL_TERMS, c_text):
        return "yes", "Legal threat or regulatory reporting requiring human legal/customer relations escalation."
    if _any_match(ESCALATION_REQUEST, c_text):
        return "yes", "Customer explicitly requested a human representative or supervisor."
    if _any_match(LOST_BAGGAGE, c_text):
        return "yes", "Lost or damaged baggage report requiring personalized claim tracing."
    if _any_match(STRONG_NEGATIVE, c_text):
        return "yes", "Severe distress or strong negative sentiment requiring human empathy and discretion."
    return "no", "Standard informational inquiry addressable by automated guidance and policy links."


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="golden/golden_unlabeled_v2.csv")
    ap.add_argument("--output", default="golden/golden_labeled_v2.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Read {len(df)} rows from {args.input}")

    rows = []
    for _, row in df.iterrows():
        c_text = str(row["customer_text_clean"])
        s_text = str(row["support_text_clean"])
        suggested = str(row["suggested_intent"])

        intent = classify_intent(c_text, s_text, suggested)
        escalate, reason = classify_escalation(c_text)
        ref_reply = s_text.strip()

        rows.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "support_tweet_id": row["support_tweet_id"],
            "customer_text_clean": c_text,
            "support_text_clean": s_text,
            "suggested_intent": suggested,
            "intent_label": intent,
            "escalate": escalate,
            "escalate_reason": reason,
            "reference_reply": ref_reply,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Labeled {len(out_df)} rows → {args.output}")
    print("\nIntent Distribution:")
    print(out_df["intent_label"].value_counts().to_string())
    print("\nEscalation Distribution:")
    print(out_df["escalate"].value_counts().to_string())


if __name__ == "__main__":
    main()
