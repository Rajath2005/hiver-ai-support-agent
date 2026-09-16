# AmericanAir Support Agent — Hiver SDE Intern Take-Home

![Project One-Pager](images/Project_One_Pager.png)

A scoped AI support agent for **one brand (AmericanAir)** from the Kaggle
"Customer Support on Twitter" dataset. It:
1. **Classifies** incoming customer tweets into a small intent set derived from the data.
2. **Drafts a grounded reply** backed by AmericanAir's historical resolutions via vector retrieval (RAG).
3. **Decides auto-handle vs. escalate**, with an explicit stated safety/legal/operational reason.

This repo deliberately does **not** try to cover multiple brands, fine-tune a
70B model, or build a production-grade vector DB. See [`decision_log.md`](decision_log.md) for
why, and [`REPORT.md`](REPORT.md) §"What's misleading about my headline number" for the
limitations of the numbers below.

---

## System Architecture & Workflow

![System Architecture](images/System_Architecture.png)

### Core System Pillars

| 1. Intent Classification | 2. Escalation & Safety | 3. Grounded Retrieval (RAG) |
| :---: | :---: | :---: |
| ![Intent Classification](images/Intent_Classification.png) | ![Escalation Decision](images/Escalation_Decision.png) | ![RAG Retrieval](images/RAG.png) |
| 8 data-derived intent clusters from query embeddings | High-precision rules for medical, legal & stranded passengers | Dense vector search (`all-MiniLM-L6-v2`) over past resolutions |

---

## Setup

```bash
pip install -r requirements.txt
```

You need:
- The Kaggle dataset `thoughtvector/customer-support-on-twitter`
  (`twcs/twcs.csv`), placed at `data/raw/twcs.csv`.
  Get it via `kaggle datasets download -d thoughtvector/customer-support-on-twitter`
  (requires a Kaggle API token at `~/.kaggle/kaggle.json`) or manual download.
- An LLM API key exported as `LLM_API_KEY` (Anthropic Claude).
  If not set, the agent falls back to returning the top-1 retrieved reply verbatim.

---


## Reproduce the headline results (<15 min, on a subsample)

### Step 1 — Automated pipeline (no human required)

```bash
make all
```

This runs in order:

1. **Prepare data:** sample 5,000 AmericanAir conversations from `twcs.csv`
2. **Split pools:** hard 90/10 split — 4,500 rows for the retrieval index,
   500 rows held out for golden sampling (zero overlap enforced by assertion)
3. **Leakage check:** exits with code 1 if any ID overlap detected
4. **Build index:** embed 4,500 index-pool replies with `all-MiniLM-L6-v2`
5. **Discover intents:** cluster golden-pool queries → `intent_taxonomy_golden.json`
6. **Sample golden unlabeled:** stratified 200-row sample from the 500-row pool

### Step 2 — Hand-label the golden set (human required, ~45–90 min for 60+ rows)

```bash
python scripts/interactive_label.py \
    --input golden/golden_unlabeled_v2.csv \
    --output golden/golden_labeled_v2.csv \
    --intents data/processed/intent_taxonomy_golden.json
```

The CLI shows each tweet pair and waits for your typed input (intent, escalate,
reason, reference reply). Press **Ctrl+C** to pause — re-run the same command
to resume where you left off.

### Step 3 — Run agent + evaluate

```bash
# Run agent over your labeled golden set
python scripts/run_agent.py \
    --golden golden/golden_labeled_v2.csv \
    --index data/processed/reply_index \
    --output eval/predictions_v2.csv

# Verify no leakage in predictions
python scripts/check_no_leakage.py \
    --index_pool data/processed/conversations_index_pool.parquet \
    --golden_pool data/processed/conversations_golden_pool.parquet \
    --predictions eval/predictions_v2.csv

# Compute all metrics
python scripts/evaluate.py \
    --predictions eval/predictions_v2.csv \
    --golden golden/golden_labeled_v2.csv \
    --output eval/results_v2.json
```

### Step 4 — LLM judge + human agreement

```bash
# 1. Produces eval/judge_llm_scores.csv + eval/judge_human_blank.csv
python scripts/llm_judge.py --predictions eval/predictions_v2.csv

# 2. Option A: Score independently using human-proxy heuristic
python scripts/score_human_independent.py --blank eval/judge_human_blank.csv --output eval/judge_human_scores.csv

#    Option B: Score manually by filling eval/judge_human_blank.csv blind using prompts/llm_judge_rubric.txt

# 3. Compute Cohen's kappa agreement
python scripts/compute_kappa.py
```

---

## Benchmark Results vs. Baselines

![Evaluation Harness & Metrics](images/Evaluvation.png)

| Metric | Our Agent | Majority-Intent Baseline | Nearest-Neighbor Baseline |
|---|---|---|---|
| **Intent Accuracy** | **46.5%** | 35.0% | n/a |
| **Intent Macro-F1** | **0.4156** | 0.0648 | n/a |
| **Mean BLEU (reply)** | **0.0215** | n/a | 0.0135 |
| **Mean BERTScore F1 (reply)** | **0.7499** | n/a | 0.7444 |
| **Escalation Precision / Recall / F1** | **1.000 / 0.750 / 0.8571** | n/a | n/a |
| **LLM-Judge Mean Score (0–4 scale)** | **3.24 / 4.0** | n/a | n/a |
| **Judge vs. Human Agreement ($\kappa$)** | **Overall $\kappa = 0.1975$** (Tone $\kappa = 0.4595$) | n/a | n/a |

See [`REPORT.md`](REPORT.md) for detailed problem framing, methodology, data leakage audit, and failure analysis.

---

## Repo structure

```
scripts/           pipeline scripts (split_pools, build_index, run_agent, evaluate, score_human_independent, ...)
prompts/           reply-generation and LLM-judge prompt templates
golden/            golden_unlabeled_v2.csv (sampled, clean split)
                   golden_labeled_v2.csv  (produced by interactive_label.py — human-only)
eval/              predictions + metrics output
data/
  raw/             twcs.csv (gitignored — download separately)
  processed/
    conversations.parquet            (5k full sample)
    conversations_index_pool.parquet (4500 rows — index only)
    conversations_golden_pool.parquet (500 rows — golden only, disjoint)
    reply_index/                      (embeddings.npy + replies.parquet)
    intent_taxonomy_golden.json       (cluster names from golden pool)
decision_log.md    17 non-obvious decisions and why
REPORT.md          the required report (<=6 pages)
```

---

## Notes on leakage prevention

![Data Leakage Prevention & Pool Split](images/Leakage_Data_Split.png)

The original pipeline had 100% retrieval leakage (all 200 golden IDs were in
the index). This is now fixed by construction:

- [`scripts/split_pools.py`](scripts/split_pools.py) creates disjoint pools with an assertion that fails
  at write time if any ID appears in both.
- [`scripts/check_no_leakage.py`](scripts/check_no_leakage.py) re-verifies this at run time (wired into [`Makefile`](Makefile)).
- [`golden/golden_unlabeled_v2.csv`](golden/golden_unlabeled_v2.csv) was sampled exclusively from [`data/processed/conversations_golden_pool.parquet`](data/processed/conversations_golden_pool.parquet).

See [`decision_log.md`](decision_log.md) entries #14–17 and [`REPORT.md`](REPORT.md) §"What went wrong in the
first pass and how it was fixed" for full details.


