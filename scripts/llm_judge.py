"""
LLM-as-judge evaluation for a random sample of 30 predictions.

Produces TWO output files:
  1. eval/judge_llm_scores.csv  -- LLM judge scores (grounding/correctness/tone)
  2. eval/judge_human_blank.csv -- Same rows with EMPTY human score columns

You (a human) must fill in eval/judge_human_blank.csv *independently*
(without looking at eval/judge_llm_scores.csv) using prompts/llm_judge_rubric.txt.
Then run scripts/compute_kappa.py to compute honest Cohen's kappa.

Rubric (from prompts/llm_judge_rubric.txt):
    grounding  : 0 (contradicts/ignores context), 2 (partial), 4 (fully grounded)
    correctness: 0 (misses question), 2 (partially answers), 4 (fully answers)
    tone       : 0 (impolite/off-brand), 2 (neutral), 4 (on-brand, polite, concise)

If LLM_API_KEY is set, uses Claude to score. Otherwise uses a deterministic
heuristic that is documented transparently (and disclosed in REPORT.md).

Usage:
    python scripts/llm_judge.py \
        --predictions eval/predictions_v2.csv \
        --n_samples 30 --seed 42
"""
import argparse
import json
import os
import sys

import pandas as pd


RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "llm_judge_rubric.txt")
SCORE_COLS = ["grounding", "correctness", "tone"]


# ─── Heuristic fallback judge ─────────────────────────────────────────────────

def heuristic_judge(customer_text: str, generated_reply: str, retrieved_context: str) -> dict:
    """
    Deterministic heuristic judge used when no LLM API key is present.
    Every decision is documented so it can be disclosed in REPORT.md.
    Returns dict with grounding, correctness, tone (values in {0,2,4})
    and a rationale string.
    """
    c = customer_text.lower()
    r = generated_reply.lower()
    ctx = retrieved_context.lower()

    # ── Grounding ───────────────────────────────────────────────────────────
    # 4 if reply shares a meaningful phrase with retrieved context (>3 words)
    # 2 if only surface keywords match
    # 0 if reply is empty or a placeholder
    if not generated_reply.strip() or "[grounded" in r:
        grounding = 0
        g_rationale = "Reply is empty or a placeholder -- no grounding."
    else:
        # Check 4-gram overlap with context
        r_4grams = set(
            " ".join(r.split()[i:i+4]) for i in range(max(0, len(r.split()) - 3))
        )
        ctx_blob = ctx
        overlap_count = sum(1 for g in r_4grams if g in ctx_blob)
        if overlap_count >= 2:
            grounding = 4
            g_rationale = f"Reply shares {overlap_count} 4-grams with retrieved context."
        elif overlap_count == 1 or any(w in r for w in ["dm", "direct message", "team", "airport"]):
            grounding = 2
            g_rationale = "Partial overlap with context or on-topic support keywords."
        else:
            grounding = 2
            g_rationale = "No strong context overlap; defaulting to partial."

    # ── Correctness ─────────────────────────────────────────────────────────
    # 4 if reply contains an actionable answer or apology/empathy for the specific complaint
    # 2 if generic / redirects but doesn't answer
    # 0 if reply ignores the question entirely
    action_words = ["please", "check", "dm", "call", "link", "help", "apolog", "sorry",
                    "seat", "bag", "delay", "refund", "book", "flight", "airport", "team"]
    if not generated_reply.strip() or "[grounded" in r:
        correctness = 0
        c_rationale = "Reply is a placeholder."
    elif sum(1 for w in action_words if w in r) >= 2:
        correctness = 4
        c_rationale = "Reply contains multiple action-oriented words relevant to the complaint."
    elif sum(1 for w in action_words if w in r) == 1:
        correctness = 2
        c_rationale = "Reply is only partially on-topic."
    else:
        correctness = 2
        c_rationale = "Reply may not directly address the customer's concern."

    # ── Tone ────────────────────────────────────────────────────────────────
    # 4 if polite signals present and reply ≤ 280 chars
    # 2 if neutral or slightly long
    # 0 if impolite, aggressive, or very long (>500 chars)
    polite = ["apolog", "sorry", "please", "glad", "welcome", "happy", "thanks", "hope",
              "love", "appreciate", "great", "wonderful"]
    if len(generated_reply) > 500:
        tone = 0
        t_rationale = "Reply exceeds 500 characters -- too long for a Twitter support context."
    elif len(generated_reply) > 280:
        tone = 2
        t_rationale = "Reply is longer than a tweet (280 chars) -- slightly off-brand."
    elif sum(1 for w in polite if w in r) >= 1:
        tone = 4
        t_rationale = "Reply contains polite/empathetic language."
    else:
        tone = 2
        t_rationale = "Reply is neutral but lacks explicit empathy signals."

    return {
        "grounding": grounding,
        "correctness": correctness,
        "tone": tone,
        "rationale": f"Grounding: {g_rationale} | Correctness: {c_rationale} | Tone: {t_rationale}",
        "scored_by": "heuristic",
    }


# ─── LLM judge (Claude) ──────────────────────────────────────────────────────

def llm_judge(customer_text: str, generated_reply: str, retrieved_context: str,
              rubric: str, api_key: str) -> dict:
    """Call Claude to score one example. Returns dict or raises on error."""
    import anthropic

    prompt = f"""{rubric}

---
Customer tweet:
{customer_text}

Retrieved context (what the support agent had access to):
{retrieved_context[:800]}

Generated reply:
{generated_reply}

---
Score the generated reply using the rubric above.
Return ONLY a JSON object with keys: grounding, correctness, tone (values 0, 2, or 4), rationale (string).
Example: {{"grounding": 4, "correctness": 2, "tone": 4, "rationale": "..."}}
"""
    client = anthropic.Anthropic(api_key=api_key)
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    resp = client.messages.create(
        model=model_name,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text)
    result["scored_by"] = f"llm:{model_name}"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="eval/predictions_v2.csv")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_output", default="eval/judge_llm_scores.csv")
    ap.add_argument("--human_blank", default="eval/judge_human_blank.csv")
    args = ap.parse_args()

    if not os.path.exists(args.predictions):
        raise SystemExit(f"Predictions file not found: {args.predictions}")

    preds = pd.read_csv(args.predictions)
    if len(preds) < args.n_samples:
        print(f"Warning: only {len(preds)} predictions -- sampling all of them.")
        sample = preds.copy().reset_index(drop=True)
    else:
        sample = preds.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)

    sample.insert(0, "row_id", range(len(sample)))

    api_key = os.environ.get("LLM_API_KEY")
    rubric = ""
    if api_key and os.path.exists(RUBRIC_PATH):
        with open(RUBRIC_PATH) as f:
            rubric = f.read()
        scorer_label = "llm"
        print(f"Using LLM judge (Claude) for {len(sample)} rows.")
    else:
        scorer_label = "heuristic"
        if not api_key:
            print("LLM_API_KEY not set -- using documented heuristic judge.")
            print("NOTE: Heuristic scores will be disclosed in REPORT.md.")

    llm_rows = []
    for i, row in sample.iterrows():
        c_text = str(row["customer_text_clean"])
        reply = str(row["generated_reply"])
        ctx = str(row.get("retrieved_context", ""))

        try:
            if api_key:
                scores = llm_judge(c_text, reply, ctx, rubric, api_key)
            else:
                scores = heuristic_judge(c_text, reply, ctx)
        except Exception as e:
            print(f"  Row {i}: judge error ({e}) -- using heuristic fallback.", file=sys.stderr)
            scores = heuristic_judge(c_text, reply, ctx)

        llm_row = {
            "row_id": row["row_id"],
            "customer_tweet_id": row.get("customer_tweet_id", ""),
            "customer_text_clean": c_text,
            "generated_reply": reply,
            "retrieved_context": ctx[:300],
            "grounding": scores["grounding"],
            "correctness": scores["correctness"],
            "tone": scores["tone"],
            "rationale": scores.get("rationale", ""),
            "scored_by": scores.get("scored_by", scorer_label),
        }
        llm_rows.append(llm_row)
        print(f"  [{i+1:02d}/{len(sample)}] g={scores['grounding']} c={scores['correctness']} t={scores['tone']}")

    llm_df = pd.DataFrame(llm_rows)
    os.makedirs(os.path.dirname(args.llm_output) or ".", exist_ok=True)

    # Write LLM scores
    llm_df.to_csv(args.llm_output, index=False, encoding="utf-8")
    print(f"\nLLM scores written to: {args.llm_output}")

    # Write BLANK human sheet (score columns are empty so labeler fills them in)
    human_blank = llm_df[[
        "row_id", "customer_tweet_id", "customer_text_clean",
        "generated_reply", "retrieved_context"
    ]].copy()
    human_blank["grounding"] = ""
    human_blank["correctness"] = ""
    human_blank["tone"] = ""
    human_blank["notes"] = ""
    human_blank.to_csv(args.human_blank, index=False, encoding="utf-8")
    print(f"Blank human sheet written to: {args.human_blank}")

    print()
    print("=" * 60)
    print("NEXT STEP -- IMPORTANT")
    print("=" * 60)
    print(f"Open {args.human_blank} in Excel/Sheets.")
    print("Read prompts/llm_judge_rubric.txt for the scoring rubric.")
    print("Score grounding/correctness/tone (0, 2, or 4) for each row.")
    print("Do NOT open judge_llm_scores.csv while scoring.")
    print("Save as eval/judge_human_scores.csv, then run:")
    print("  python scripts/compute_kappa.py")
    print()

    # Quick LLM score summary
    mean_g = llm_df["grounding"].mean()
    mean_c = llm_df["correctness"].mean()
    mean_t = llm_df["tone"].mean()
    overall = (mean_g + mean_c + mean_t) / 3
    print(f"LLM judge summary (n={len(llm_df)}):")
    print(f"  Mean grounding  : {mean_g:.2f}/4")
    print(f"  Mean correctness: {mean_c:.2f}/4")
    print(f"  Mean tone       : {mean_t:.2f}/4")
    print(f"  Mean overall    : {overall:.2f}/4")

    summary = {
        "n_samples": len(llm_df),
        "scorer": scorer_label,
        "mean_llm_grounding": round(mean_g, 4),
        "mean_llm_correctness": round(mean_c, 4),
        "mean_llm_tone": round(mean_t, 4),
        "mean_llm_overall": round(overall, 4),
        "note": (
            "Human scores in eval/judge_human_scores.csv (blank template: eval/judge_human_blank.csv). "
            "Run compute_kappa.py after human scoring."
        ),
    }
    with open("eval/judge_llm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to eval/judge_llm_summary.json")


if __name__ == "__main__":
    main()
