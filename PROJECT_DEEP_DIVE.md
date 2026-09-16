# End-to-End Deep Dive: AmericanAir AI Support Agent

This document provides a comprehensive, file-by-file technical explanation of the **AmericanAir AI Support Agent** built for the Hiver SDE Intern Take-Home Assignment. It includes complete architectural breakdowns, code blocks extracted directly from the codebase, mathematical foundations, data leakage prevention strategies, and evaluation methodologies.

---

## 📋 Table of Contents

1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Data Preparation & Leakage Prevention](#2-data-preparation--leakage-prevention)
   - [`scripts/prepare_data.py`](#scriptsprepare_datapy)
   - [`scripts/split_pools.py`](#scriptssplit_poolspy)
   - [`scripts/check_no_leakage.py`](#scriptscheck_no_leakagepy)
3. [Vector Indexing & Intent Discovery](#3-vector-indexing--intent-discovery)
   - [`scripts/build_index.py`](#scriptsbuild_indexpy)
   - [`scripts/discover_intents.py`](#scriptsdiscover_intentspy)
4. [Safety & Escalation Rules Engine](#4-safety--escalation-rules-engine)
   - [`scripts/escalation.py`](#scriptsescalationpy)
5. [Core Agent Execution Engine](#5-core-agent-execution-engine)
   - [`scripts/run_agent.py`](#scriptsrun_agentpy)
6. [Evaluation Harness & Human Agreement](#6-evaluation-harness--human-agreement)
   - [`scripts/evaluate.py`](#scriptsevaluatepy)
   - [`scripts/llm_judge.py`](#scriptsllm_judgepy)
   - [`scripts/score_human_independent.py`](#scriptsscore_human_independentpy)
   - [`scripts/compute_kappa.py`](#scriptscompute_kappapy)
7. [Benchmark Results & Baseline Comparisons](#7-benchmark-results--baseline-comparisons)
8. [Reproducibility & Command Execution](#8-reproducibility--command-execution)

---

## 1. Executive Summary & System Architecture

The AI Support Agent is engineered for a single high-volume domain brand: **AmericanAir**, extracted from the Kaggle *Customer Support on Twitter* dataset (~3M tweets).

```
 ┌──────────────────────┐
 │ Customer Tweet       │
 └──────────┬───────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. Intent Classification                                    │
 │    - Embed query via `all-MiniLM-L6-v2`                     │
 │    - Match nearest centroid in 8 data-derived intents        │
 └──────────┬──────────────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 2. Grounded RAG Retrieval                                   │
 │    - Dot product matrix multiplication (@) over 4,500 index  │
 │    - Retrieve top-1 historical AmericanAir resolution        │
 └──────────┬──────────────────────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 3. Explainable Escalation Check                             │
 │    - Rule engine for medical, legal, stranded, explicit     │
 │    - Outputs boolean `escalate` + human-readable reason       │
 └──────────┬──────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Final Draft / Action │
 └──────────────────────┘
```

---

## 2. Data Preparation & Leakage Prevention

### `scripts/prepare_data.py`
Filters raw Twitter conversations to pair customer inquiries with brand replies from `@AmericanAir`, cleans text (removes usernames, links, extra spaces), and outputs `data/processed/conversations.parquet`.

```python
def prepare_data(input_csv: str, brand: str, sample_size: int, output_parquet: str, seed: int):
    df = pd.read_csv(input_csv)
    
    # Identify tweets originating from or sent to the specified brand
    brand_tweets = df[df['author_id'].astype(str).str.lower() == brand.lower()]
    
    # Pair customer tweet with brand response via in_response_to_tweet_id
    merged = df.merge(
        brand_tweets, 
        left_on='tweet_id', 
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_support')
    )
    
    # Text cleaning
    merged['customer_text_clean'] = merged['text_customer'].apply(clean_text)
    merged['support_text_clean'] = merged['text_support'].apply(clean_text)
    
    # Subsample to specified size
    sampled = merged.sample(n=sample_size, random_state=seed)
    sampled.to_parquet(output_parquet, index=False)
```

---

### `scripts/split_pools.py`
Enforces a strict 90/10 disjoint split (4,500 rows for the index pool, 500 rows for the golden pool) with a hard runtime assertion to guarantee **0 data leakage**.

```python
def split_pools(input_parquet: str, output_dir: str, seed: int):
    df = pd.read_parquet(input_parquet)
    
    # 90% Index Pool, 10% Golden Pool
    golden_pool = df.sample(frac=0.10, random_state=seed)
    index_pool = df.drop(golden_pool.index)
    
    # HARD LEAKAGE ASSERTION
    index_ids = set(index_pool['customer_tweet_id'])
    golden_ids = set(golden_pool['customer_tweet_id'])
    assert len(index_ids.intersection(golden_ids)) == 0, "Data leakage detected between index and golden pools!"
    
    index_pool.to_parquet(os.path.join(output_dir, "conversations_index_pool.parquet"), index=False)
    golden_pool.to_parquet(os.path.join(output_dir, "conversations_golden_pool.parquet"), index=False)
```

---

### `scripts/check_no_leakage.py`
Runtime verification script invoked during automated pipelines (`make all`, `make run`) to verify 0 ID overlap across index, golden, and prediction files.

```python
def check_leakage(index_path: str, golden_path: str, predictions_path: str = None):
    index_df = pd.read_parquet(index_path)
    golden_df = pd.read_parquet(golden_path)
    
    index_ids = set(index_df["customer_tweet_id"])
    golden_ids = set(golden_df["customer_tweet_id"])
    
    overlap = index_ids.intersection(golden_ids)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: {len(overlap)} IDs overlap between index and golden pool!")
    print("✅ Leakage check PASSED: 0 ID overlap confirmed.")
```

---

## 3. Vector Indexing & Intent Discovery

### `scripts/build_index.py`
Embeds 4,500 historical support replies using `sentence-transformers/all-MiniLM-L6-v2`. L2-normalizes the vectors so dot products (`@`) compute exact cosine similarity.

```python
from sentence_transformers import SentenceTransformer

def build_index(conversations_path: str, output_dir: str):
    df = pd.read_parquet(conversations_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate normalized 384-dimensional dense vectors
    embeddings = model.encode(
        df["support_text_clean"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    df[["customer_tweet_id", "support_text_clean", "customer_text_clean"]].to_parquet(
        os.path.join(output_dir, "replies.parquet"), index=False
    )
```

---

### `scripts/discover_intents.py`
Runs K-Means clustering ($K=8$) on customer query embeddings to discover data-derived intent categories rather than using a hardcoded or assumed FAQ taxonomy.

```python
from sklearn.cluster import KMeans

def discover_intents(conversations_path: str, n_clusters: int, seed: int, output_json: str):
    df = pd.read_parquet(conversations_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    embeddings = model.encode(df["customer_text_clean"].tolist(), normalize_embeddings=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(embeddings)
    
    # Assign names to the 8 cluster centroids
    taxonomy = {}
    for cluster_id in range(n_clusters):
        cluster_samples = df[kmeans.labels_ == cluster_id]["customer_text_clean"].head(5).tolist()
        taxonomy[f"intent_cluster_{cluster_id}"] = {
            "center_vector_idx": int(cluster_id),
            "sample_texts": cluster_samples
        }
        
    with open(output_json, "w") as f:
        json.dump(taxonomy, f, indent=2)
```

---

## 4. Safety & Escalation Rules Engine

### `scripts/escalation.py`
A transparent, explainable rule engine that checks incoming customer queries for medical emergencies, legal threats, stranded passengers, and explicit requests for human agents.

```python
class EscalationEngine:
    TRIGGERS = {
        "medical": ["medical", "hospital", "injured", "doctor", "wheelchair", "hurt"],
        "legal": ["lawyer", "sue", "legal action", "attorney", "court"],
        "stranded": ["stranded", "sleeping at airport", "overnight", "no hotel"],
        "explicit": ["human", "agent", "supervisor", "representative", "manager"]
    }

    def evaluate(self, customer_text: str) -> tuple[bool, str]:
        text_lower = customer_text.lower()
        
        for category, keywords in self.TRIGGERS.items():
            for kw in keywords:
                if kw in text_lower:
                    return True, f"Triggered {category} rule keyword: '{kw}'"
                    
        return False, "Handled by automated agent (no escalation triggers detected)"
```

---

## 5. Core Agent Execution Engine

### `scripts/run_agent.py`
End-to-end pipeline combining intent assignment, vector retrieval (RAG), draft reply synthesis, and escalation decision.

```python
class SupportAgent:
    def __init__(self, index_dir: str):
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.replies_df = pd.read_parquet(os.path.join(index_dir, "replies.parquet"))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.escalation = EscalationEngine()

    def process_query(self, customer_text: str) -> dict:
        # 1. Embed incoming query
        query_vec = self.model.encode([customer_text], normalize_embeddings=True)[0]
        
        # 2. Vector Retrieval (1-line matrix dot product)
        similarities = self.embeddings @ query_vec
        top_idx = int(similarities.argmax())
        retrieved_reply = self.replies_df.iloc[top_idx]["support_text_clean"]
        
        # 3. Escalation Check
        should_escalate, rationale = self.escalation.evaluate(customer_text)
        
        return {
            "predicted_reply": retrieved_reply,
            "escalate": should_escalate,
            "escalation_reason": rationale
        }
```

---

## 6. Evaluation Harness & Human Agreement

### `scripts/evaluate.py`
Computes key evaluation metrics against the human-labeled golden set: Intent Accuracy, Intent Macro-F1, BLEU Score, BERTScore F1, and Escalation Precision/Recall/F1.

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from bert_score import score as bert_score_eval

def evaluate_predictions(predictions_csv: str, golden_csv: str, output_json: str):
    preds = pd.read_csv(predictions_csv)
    golden = pd.read_csv(golden_csv)
    merged = preds.merge(golden, on="customer_tweet_id")

    # Intent Metrics
    intent_acc = accuracy_score(merged["intent_gold"], merged["intent_pred"])
    intent_f1 = f1_score(merged["intent_gold"], merged["intent_pred"], average="macro")

    # Reply Quality (BERTScore F1)
    _, _, f1_scores = bert_score_eval(
        merged["generated_reply"].tolist(),
        merged["reference_reply"].tolist(),
        model_type="distilbert-base-uncased",
        num_layers=5
    )

    # Escalation Precision/Recall/F1 on Positive Class
    esc_true = (merged["escalate_gold"] == "yes")
    esc_pred = (merged["escalate_pred"] == True)
    
    esc_precision = precision_score(esc_true, esc_pred, pos_label=True)
    esc_recall = recall_score(esc_true, esc_pred, pos_label=True)
    esc_f1 = f1_score(esc_true, esc_pred, pos_label=True)

    results = {
        "intent_accuracy": float(intent_acc),
        "intent_macro_f1": float(intent_f1),
        "mean_bertscore_f1": float(f1_scores.mean().item()),
        "escalation_precision_positive": float(esc_precision),
        "escalation_recall_positive": float(esc_recall),
        "escalation_f1_positive": float(esc_f1)
    }
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
```

---

### `scripts/compute_kappa.py`
Computes inter-rater agreement (Cohen's Kappa $\kappa$) between LLM judge scores and human reference scores.

```python
from sklearn.metrics import cohen_kappa_score

def compute_kappa(llm_csv: str, human_csv: str):
    llm_df = pd.read_csv(llm_csv)
    human_df = pd.read_csv(human_csv)
    
    kappa_grounding = cohen_kappa_score(llm_df["grounding"], human_df["grounding"])
    kappa_correctness = cohen_kappa_score(llm_df["correctness"], human_df["correctness"])
    kappa_tone = cohen_kappa_score(llm_df["tone"], human_df["tone"])
    
    overall_kappa = (kappa_grounding + kappa_correctness + kappa_tone) / 3.0
    return {
        "kappa_grounding": kappa_grounding,
        "kappa_correctness": kappa_correctness,
        "kappa_tone": kappa_tone,
        "overall_kappa": overall_kappa
    }
```

---

## 7. Benchmark Results & Baseline Comparisons

| Metric | Our Agent | Majority-Intent Baseline | Nearest-Neighbor Baseline |
|---|---|---|---|
| **Intent Accuracy** | **46.5%** | 35.0% | n/a |
| **Intent Macro-F1** | **0.4156** | 0.0648 | n/a |
| **Mean BLEU (reply)** | **0.0215** | n/a | 0.0135 |
| **Mean BERTScore F1 (reply)** | **0.7499** | n/a | 0.7444 |
| **Escalation Precision / Recall / F1** | **1.000 / 0.750 / 0.8571** | n/a | n/a |
| **LLM-Judge Mean Score (0–4 scale)** | **3.24 / 4.0** | n/a | n/a |
| **Judge vs. Human Agreement ($\kappa$)** | **Overall $\kappa = 0.1975$** (Tone $\kappa = 0.4595$) | n/a | n/a |

---

## 8. Reproducibility & Command Execution

To execute the end-to-end pipeline using [`Makefile`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/Makefile):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data preparation, pool splitting, leakage check, indexing, and intent discovery
make all

# 3. Execute agent predictions over golden set
make run

# 4. Evaluate metrics harness
make evaluate

# 5. Run LLM judge and compute Cohen's Kappa agreement
make judge
```
