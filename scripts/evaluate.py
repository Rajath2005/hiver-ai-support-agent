"""
Evaluation harness producing:
  - intent accuracy/F1 vs. two baselines (majority-intent and nearest-neighbour reply)
  - reply quality: BLEU (rough) and BERTScore vs. human reference
  - escalation: precision / recall / F1 on the positive (escalate=yes) class
  - judge/human agreement summary (from eval/kappa_results.json if present)

Usage:
    python evaluate.py \
        --predictions eval/predictions_v2.csv \
        --golden golden/golden_labeled_v2.csv \
        --output eval/results_v2.json
"""
import os
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def majority_intent_baseline(golden: pd.DataFrame) -> str:
    return Counter(golden["intent_label"]).most_common(1)[0][0]


def bleu_score(pred: str, ref: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    return sentence_bleu(
        [ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1
    )


def bertscore_f1(preds, refs):
    from bert_score import score

    _, _, f1 = score(preds, refs, model_type="distilbert-base-uncased", num_layers=5, verbose=False)
    return f1.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions)
    golden = pd.read_csv(args.golden)
    merged = preds.merge(golden, on="customer_tweet_id", suffixes=("_pred", "_gold"))

    results = {}

    # --- Intent classification ---
    has_labels = merged["intent_label"].astype(str).str.len() > 0
    labeled = merged[has_labels]
    if len(labeled):
        y_true = labeled["intent_label"]
        y_pred = labeled["predicted_intent"]
        results["intent_accuracy"] = float(accuracy_score(y_true, y_pred))
        results["intent_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        maj = majority_intent_baseline(golden)
        y_maj = [maj] * len(y_true)
        results["baseline_majority_intent_accuracy"] = float((y_true == maj).mean())
        results["baseline_majority_intent_macro_f1"] = float(f1_score(y_true, y_maj, average="macro", zero_division=0))

    # --- Reply quality (Agent) ---
    has_ref = merged["reference_reply"].astype(str).str.len() > 0
    ref_rows = merged[has_ref]
    if len(ref_rows):
        bleus = [
            bleu_score(str(p), str(r))
            for p, r in zip(ref_rows["generated_reply"], ref_rows["reference_reply"])
        ]
        results["mean_bleu"] = float(sum(bleus) / len(bleus))
        try:
            results["mean_bertscore_f1"] = float(bertscore_f1(
                ref_rows["generated_reply"].astype(str).tolist(),
                ref_rows["reference_reply"].astype(str).tolist(),
            ))
        except Exception as e:
            results["mean_bertscore_f1_error"] = str(e)

    # --- Nearest Neighbor Reply Baseline ---
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    c_col = "customer_text_clean_gold" if "customer_text_clean_gold" in ref_rows.columns else "customer_text_clean"
    s_col = "support_text_clean_gold" if "support_text_clean_gold" in ref_rows.columns else "support_text_clean"
    gold_embs = emb_model.encode(ref_rows[c_col].tolist(), normalize_embeddings=True)
    nn_replies = []
    for i in range(len(ref_rows)):
        sims = gold_embs @ gold_embs[i]
        sims[i] = -1.0  # exclude self
        nn_idx = sims.argmax()
        nn_replies.append(ref_rows.iloc[nn_idx][s_col])
    
    nn_bleus = [
        bleu_score(str(p), str(r))
        for p, r in zip(nn_replies, ref_rows["reference_reply"])
    ]
    results["baseline_nn_mean_bleu"] = float(sum(nn_bleus) / len(nn_bleus))
    try:
        results["baseline_nn_mean_bertscore_f1"] = float(bertscore_f1(
            nn_replies,
            ref_rows["reference_reply"].astype(str).tolist(),
        ))
    except Exception as e:
        results["baseline_nn_mean_bertscore_f1_error"] = str(e)

    # --- Escalation: accuracy + precision/recall/F1 on positive class ---
    # NOTE: accuracy is misleading on imbalanced labels (most tweets don't
    # need escalation). P/R/F1 on the positive class is the honest metric.
    if "escalate_pred" in merged.columns and "escalate_gold" in merged.columns:
        esc_true = (merged["escalate_gold"].astype(str).str.lower() == "yes")
        esc_pred = (merged["escalate_pred"].astype(str).str.lower().isin(["true", "yes"]))
        results["escalation_accuracy"] = float(accuracy_score(esc_true, esc_pred))
        results["escalation_macro_f1"] = float(f1_score(esc_true, esc_pred, average="macro", zero_division=0))
        results["escalation_precision_positive"] = float(
            precision_score(esc_true, esc_pred, pos_label=True, zero_division=0)
        )
        results["escalation_recall_positive"] = float(
            recall_score(esc_true, esc_pred, pos_label=True, zero_division=0)
        )
        results["escalation_f1_positive"] = float(
            f1_score(esc_true, esc_pred, pos_label=True, zero_division=0)
        )
        n_pos_true = int(esc_true.sum())
        n_pos_pred = int(esc_pred.sum())
        results["escalation_n_true_positive"] = n_pos_true
        results["escalation_n_predicted_positive"] = n_pos_pred
        results["escalation_note"] = (
            f"Positive class (escalate=yes): {n_pos_true}/{len(esc_true)} true, "
            f"{n_pos_pred}/{len(esc_pred)} predicted. "
            "Accuracy is misleading on this imbalanced label; use F1_positive."
        )

    # --- Judge/human agreement (from compute_kappa.py output) ---
    kappa_path = os.path.join(os.path.dirname(args.output), "kappa_results.json")
    llm_summary_path = os.path.join(os.path.dirname(args.output), "judge_llm_summary.json")
    if os.path.exists(kappa_path):
        with open(kappa_path) as jf:
            results["judge_human_agreement"] = json.load(jf)
    elif os.path.exists(llm_summary_path):
        with open(llm_summary_path) as jf:
            results["judge_llm_summary"] = json.load(jf)
        results["note_judge_human_agreement"] = (
            "Human scores not yet provided. Fill eval/judge_human_blank.csv "
            "then run scripts/compute_kappa.py."
        )
    else:
        results["note_judge_human_agreement"] = (
            "Run scripts/llm_judge.py, fill eval/judge_human_blank.csv, "
            "then run scripts/compute_kappa.py."
        )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
