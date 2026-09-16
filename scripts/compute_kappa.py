"""
Compute honest Cohen's kappa between LLM-judge scores and human scores.

Reads two separate CSVs:
  - eval/judge_llm_scores.csv  (produced by llm_judge.py)
  - eval/judge_human_scores.csv (filled in BY A HUMAN, blind to LLM scores)

and merges them on `row_id` to compute per-dimension and overall kappa.

Usage:
    python scripts/compute_kappa.py \
        --llm_scores eval/judge_llm_scores.csv \
        --human_scores eval/judge_human_scores.csv \
        --output eval/kappa_results.json
"""
import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score


SCORE_COLS = ["grounding", "correctness", "tone"]
VALID_SCORES = {0, 2, 4}


def validate_scores(df: pd.DataFrame, label: str) -> bool:
    ok = True
    for col in SCORE_COLS:
        if col not in df.columns:
            print(f"ERROR [{label}]: missing column '{col}'", file=sys.stderr)
            ok = False
            continue
        bad = df[col].dropna()[~df[col].dropna().isin(VALID_SCORES)]
        if len(bad):
            print(
                f"ERROR [{label}]: column '{col}' has values outside {{0,2,4}}: "
                f"{bad.unique().tolist()}",
                file=sys.stderr,
            )
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_scores", default="eval/judge_llm_scores.csv")
    ap.add_argument("--human_scores", default="eval/judge_human_scores.csv")
    ap.add_argument("--output", default="eval/kappa_results.json")
    args = ap.parse_args()

    llm_df = pd.read_csv(args.llm_scores)
    human_df = pd.read_csv(args.human_scores)

    print(f"LLM scores  : {len(llm_df)} rows from {args.llm_scores}")
    print(f"Human scores: {len(human_df)} rows from {args.human_scores}")

    # Check for empty human scores (common mistake -- labeler forgot to fill in)
    human_empty = human_df[SCORE_COLS].isnull().all(axis=1).sum()
    if human_empty > 0:
        print(
            f"WARNING: {human_empty} rows in human scores have all-null score columns. "
            "Fill these in before computing kappa.",
            file=sys.stderr,
        )

    ok_llm = validate_scores(llm_df, "llm_scores")
    ok_human = validate_scores(human_df, "human_scores")
    if not (ok_llm and ok_human):
        print("\nFix the above errors before running compute_kappa.py.", file=sys.stderr)
        sys.exit(1)

    # Merge on row_id
    if "row_id" not in llm_df.columns or "row_id" not in human_df.columns:
        print(
            "ERROR: both CSVs must have a 'row_id' column for reliable merging.",
            file=sys.stderr,
        )
        sys.exit(1)

    merged = llm_df.merge(human_df, on="row_id", suffixes=("_llm", "_human"))
    n = len(merged)
    if n == 0:
        print("ERROR: no rows matched on row_id -- check that both files cover the same sample.", file=sys.stderr)
        sys.exit(1)
    print(f"Matched {n} rows on row_id.\n")

    results = {"n_samples": n}
    for col in SCORE_COLS:
        llm_col = f"{col}_llm"
        human_col = f"{col}_human"
        kappa = cohen_kappa_score(merged[llm_col].astype(int), merged[human_col].astype(int))
        results[f"cohen_kappa_{col}"] = round(float(kappa), 4)
        print(f"  {col:12s}: kappa = {kappa:.4f}")

    # Overall: bin the average score to nearest {0, 2, 4}
    llm_avg = merged[[f"{c}_llm" for c in SCORE_COLS]].mean(axis=1)
    human_avg = merged[[f"{c}_human" for c in SCORE_COLS]].mean(axis=1)
    binned_llm = (llm_avg / 2).round() * 2
    binned_human = (human_avg / 2).round() * 2
    kappa_overall = cohen_kappa_score(binned_llm.astype(int), binned_human.astype(int))
    results["cohen_kappa_overall"] = round(float(kappa_overall), 4)
    print(f"  {'overall':12s}: kappa = {kappa_overall:.4f}")

    # Interpretation guide
    if kappa_overall < 0.2:
        interp = "Slight agreement -- judge and human are mostly disagreeing."
    elif kappa_overall < 0.4:
        interp = "Fair agreement."
    elif kappa_overall < 0.6:
        interp = "Moderate agreement -- typical for open-ended NLG evaluation."
    elif kappa_overall < 0.8:
        interp = "Substantial agreement -- judge is quite reliable."
    else:
        interp = "Almost perfect agreement."
    results["interpretation"] = interp
    print(f"\n  -> {interp}")

    # Disagreement examples (for REPORT.md)
    disagree_rows = merged[merged[f"grounding_llm"] != merged[f"grounding_human"]]
    examples = []
    for _, r in disagree_rows.head(3).iterrows():
        ex = {
            "row_id": int(r["row_id"]),
            "llm_grounding": int(r["grounding_llm"]),
            "human_grounding": int(r["grounding_human"]),
        }
        if "customer_text_clean" in r:
            ex["customer_text"] = r["customer_text_clean"]
        if "generated_reply" in r:
            ex["generated_reply"] = r["generated_reply"]
        examples.append(ex)
    results["disagreement_examples_grounding"] = examples

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
