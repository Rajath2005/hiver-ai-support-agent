# Live Interview Code Modifications & Cheat Sheet

This document contains **10 realistic live coding tasks** that interviewers at Hiver may ask you to perform during your live technical round, complete with target file locations, exact line-by-line diffs, code snippets, and terminal execution commands.

---

## 🛠️ Task Index

1. [Task 1: Change Intent Clusters (e.g. from 8 to 12 intents)](#task-1-change-intent-clusters)
2. [Task 2: Add a New Escalation Category (e.g. "Baggage Loss" or "Refund Claim")](#task-2-add-a-new-escalation-category)
3. [Task 3: Retrieve Top-K Matches Instead of Top-1 (RAG Multi-Context)](#task-3-retrieve-top-k-matches-instead-of-top-1)
4. [Task 4: Add a Similarity Threshold Guardrail (Fallback Reply for Low Confidence)](#task-4-add-a-similarity-threshold-guardrail)
5. [Task 5: Change the Data Split Ratio (e.g. 80/20 instead of 90/10)](#task-5-change-the-data-split-ratio)
6. [Task 6: Switch Target Brand (e.g. from `@AmericanAir` to `@SouthwestAir`)](#task-6-switch-target-brand)
7. [Task 7: Change the Sentence Embedding Model (e.g. to `all-mpnet-base-v2`)](#task-7-change-the-sentence-embedding-model)
8. [Task 8: Add Per-Class Intent Precision/Recall Metrics](#task-8-add-per-class-intent-precisionrecall-metrics)
9. [Task 9: Add Exclude-Self Logic in RAG Retrieval](#task-9-add-exclude-self-logic-in-rag-retrieval)
10. [Task 10: Customize LLM System Prompt for Tone / Formatting](#task-10-customize-llm-system-prompt)

---

## Task 1: Change Intent Clusters

**Scenario:** *"The 8 intent clusters look a bit coarse. Can you change the pipeline live to extract 12 intent clusters instead?"*

- **Target File:** [`scripts/discover_intents.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/discover_intents.py) / [`Makefile`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/Makefile)

### Exact Code Edit:
In `Makefile` line 24:
```diff
discover:
 	python scripts/discover_intents.py \
 		--conversations data/processed/conversations_golden_pool.parquet \
-		--n_clusters 8 --seed 42 \
+		--n_clusters 12 --seed 42 \
 		--output data/processed/intent_taxonomy_golden.json
```

### Terminal Command:
```bash
python scripts/discover_intents.py --conversations data/processed/conversations_golden_pool.parquet --n_clusters 12 --seed 42 --output data/processed/intent_taxonomy_golden.json
```

---

## Task 2: Add a New Escalation Category

**Scenario:** *"Can you add an escalation rule for baggage loss or claim disputes so that tweets mentioning 'lost bag' automatically escalate to a human agent?"*

- **Target File:** [`scripts/escalation.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/escalation.py)

### Exact Code Edit:
In `scripts/escalation.py` line 14:
```diff
 ESCALATION_TRIGGERS = {
     'medical': ['medical', 'hospital', 'injured', 'hurt', 'wheelchair'],
     'legal': ['lawyer', 'sue', 'legal', 'attorney'],
     'stranded': ['stranded', 'overnight', 'sleeping on floor'],
-    'explicit': ['agent', 'human', 'supervisor', 'representative']
+    'explicit': ['agent', 'human', 'supervisor', 'representative'],
+    'baggage_loss': ['lost bag', 'lost baggage', 'stolen luggage', 'missing bag']
 }
```

### Terminal Test Verification:
```bash
python -c "from scripts.escalation import EscalationEngine; engine = EscalationEngine(); print(engine.evaluate('My luggage is lost!'))"
```

---

## Task 3: Retrieve Top-K Matches Instead of Top-1

**Scenario:** *"Currently the agent retrieves only top-1 reply. Can you modify `run_agent.py` to retrieve top-3 matching replies for multi-context RAG?"*

- **Target File:** [`scripts/run_agent.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/run_agent.py)

### Exact Code Edit:
In `scripts/run_agent.py`:
```diff
-top_idx = int(similarities.argmax())
-retrieved_reply = self.replies_df.iloc[top_idx]["support_text_clean"]
+top_k_indices = np.argsort(similarities)[-3:][::-1]
+retrieved_replies = [self.replies_df.iloc[i]["support_text_clean"] for i in top_k_indices]
+combined_context = "\n---\n".join(retrieved_replies)
```

---

## Task 4: Add a Similarity Threshold Guardrail

**Scenario:** *"If the cosine similarity between the query and top retrieved reply is below 0.50, return a default helpful fallback message instead of copying an irrelevant reply."*

- **Target File:** [`scripts/run_agent.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/run_agent.py)

### Exact Code Edit:
```python
top_idx = int(similarities.argmax())
max_sim = float(similarities[top_idx])

if max_sim < 0.50:
    retrieved_reply = "Thank you for contacting AmericanAir. Please DM us your 6-character confirmation code so our support team can assist you."
else:
    retrieved_reply = self.replies_df.iloc[top_idx]["support_text_clean"]
```

---

## Task 5: Change the Data Split Ratio

**Scenario:** *"Can you change the dataset partition from 90/10 to 80/20 (80% index, 20% golden)?"*

- **Target File:** [`scripts/split_pools.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/split_pools.py)

### Exact Code Edit:
In `scripts/split_pools.py` line 18:
```diff
-golden_pool = df.sample(frac=0.10, random_state=seed)
+golden_pool = df.sample(frac=0.20, random_state=seed)
 index_pool = df.drop(golden_pool.index)
```

### Terminal Command:
```bash
python scripts/split_pools.py --input data/processed/conversations.parquet --output_dir data/processed --seed 42
```

---

## Task 6: Switch Target Brand

**Scenario:** *"How would you re-run the pipeline for `@SouthwestAir` instead of `@AmericanAir`?"*

- **Target File:** [`Makefile`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/Makefile) / [`scripts/prepare_data.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/prepare_data.py)

### Exact Code Edit / Command:
In `Makefile`:
```diff
 split:
-	python scripts/prepare_data.py --input data/raw/twcs.csv --brand AmericanAir \
+	python scripts/prepare_data.py --input data/raw/twcs.csv --brand SouthwestAir \
 		--sample_conversations 5000 --seed 42 --output data/processed/conversations.parquet
```

---

## Task 7: Change the Sentence Embedding Model

**Scenario:** *"Swap out `all-MiniLM-L6-v2` for `all-mpnet-base-v2` across the pipeline."*

- **Target Files:** [`scripts/build_index.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/build_index.py), [`scripts/discover_intents.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/discover_intents.py), [`scripts/run_agent.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/run_agent.py)

### Exact Code Edit:
Replace model initialization string:
```diff
-SentenceTransformer("all-MiniLM-L6-v2")
+SentenceTransformer("all-mpnet-base-v2")
```

---

## Task 8: Add Per-Class Intent Precision/Recall Metrics

**Scenario:** *"Modify `evaluate.py` so it prints a full classification report for each individual intent class."*

- **Target File:** [`scripts/evaluate.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/evaluate.py)

### Exact Code Edit:
In `scripts/evaluate.py` line 67:
```python
from sklearn.metrics import classification_report

print("\n--- Detailed Per-Class Intent Report ---")
print(classification_report(y_true, y_pred, zero_division=0))
```

---

## Task 9: Add Exclude-Self Logic in RAG Retrieval

**Scenario:** *"Ensure that when evaluating, a query tweet never retrieves its own tweet ID if present in the pool."*

- **Target File:** [`scripts/run_agent.py`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/scripts/run_agent.py)

### Exact Code Edit:
```python
# Zero out similarity score for exact matching tweet ID
if "customer_tweet_id" in row:
    exact_match_mask = (self.replies_df["customer_tweet_id"] == row["customer_tweet_id"])
    similarities[exact_match_mask] = -1.0

top_idx = int(similarities.argmax())
```

---

## Task 10: Customize LLM System Prompt

**Scenario:** *"Modify the reply generation prompt template to enforce a polite professional tone ending with a support reference number."*

- **Target File:** [`prompts/reply_generation.txt`](file:///d:/Downloads/hiver-support-agent/hiver-support-agent/prompts/reply_generation.txt)

### Exact Text Edit:
```text
You are an empathetic, highly professional customer support representative for AmericanAir.
Draft a concise support reply for the customer inquiry below based ONLY on the provided historical resolution.

Guidelines:
1. Be empathetic and professional.
2. Ground all facts strictly in the historical resolution.
3. Always close with: "Case Ref: #AA-SUPPORT".
```
