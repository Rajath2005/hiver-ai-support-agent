"""
Transparent, rule-based escalation heuristic (decision_log.md #6).
Returns (should_escalate: bool, reason: str) so every decision is auditable.
"""
import re

SAFETY_TERMS = [r"\bunsafe\b", r"\bemergency\b", r"\bmedical\b", r"\bassault\b", r"\bthreat(en)?\b"]
LEGAL_TERMS = [r"\blawyer\b", r"\blawsuit\b", r"\bsue\b", r"\blegal action\b", r"\battorney\b"]
ESCALATION_REQUEST_TERMS = [r"\bsupervisor\b", r"\bmanager\b", r"\bhuman\b", r"\breal person\b"]
STRONG_NEGATIVE_TERMS = [
    r"\bdisgusting\b", r"\bterrible\b", r"\bworst\b", r"\bnever again\b",
    r"\bscam\b", r"\bfraud\b", r"\bunacceptable\b",
]


def _any_match(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def decide_escalation(customer_text: str, thread_unresolved_count: int = 0) -> tuple:
    """
    thread_unresolved_count: how many prior support replies in this thread
    the customer has already responded negatively to (0 if unknown/first turn).
    """
    if _any_match(SAFETY_TERMS, customer_text):
        return True, "Safety-related language detected; brand policy escalates safety concerns to a human."
    if _any_match(LEGAL_TERMS, customer_text):
        return True, "Legal language detected (e.g. lawyer/lawsuit); brand policy escalates legal threats to a human."
    if _any_match(ESCALATION_REQUEST_TERMS, customer_text):
        return True, "Customer explicitly requested a supervisor/human."
    if thread_unresolved_count >= 2:
        return True, f"Thread has {thread_unresolved_count} prior unresolved exchanges; auto-reply unlikely to help."
    if _any_match(STRONG_NEGATIVE_TERMS, customer_text):
        return True, "Strong negative sentiment language detected (e.g. scam/unacceptable/worst)."
    return False, "No safety, legal, explicit-escalation, or strong-negative signal found; auto-handling."
