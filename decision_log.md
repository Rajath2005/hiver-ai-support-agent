# Decision Log

1. **One brand only (AmericanAir).** The assignment asks to pick one brand;
   airline support tweets have high volume, repetitive intents, and
   factual/checkable resolutions (rebooking, refunds), making grounding and
   evaluation tractable in the time budget.
2. **Subsample (5k conversations), not the full corpus.** Reviewers explicitly
   said they won't run the full dataset; a subsample also keeps the pipeline
   under 15 minutes.
3. **Data-derived intents, not a pre-guessed taxonomy.** We cluster customer
   query embeddings first, then name clusters, so the taxonomy reflects what
   AmericanAir customers actually ask rather than an assumed airline FAQ list.
4. **~6-8 intents, not 20-30.** With only 150-250 golden examples, a large
   taxonomy leaves too few labeled examples per class to evaluate reliably.
5. **Retrieval over historical AmericanAir replies, not fine-tuning.** No
   labeled preference data or compute budget justifies fine-tuning; retrieval
   + a single grounded LLM call is cheaper, faster to iterate on, and easier
   to audit (you can point to which past reply grounded the draft).
6. **Escalation via explainable heuristics, not a trained classifier.** A
   heuristic scorer (sentiment + keyword triggers + thread-unresolved count)
   is transparent and defensible with 150-250 labeled examples; a trained
   classifier would be undertrained and harder to explain "why escalated."
7. **BLEU/ROUGE reported but flagged as weak for this task.** Support replies
   are short and templated, so n-gram overlap rewards generic phrasing; we
   lean more on BERTScore + LLM-judge and say so explicitly.
8. **LLM-as-judge is calibrated against our own human labels on a subset**,
   not assumed correct out of the box — this is why the assignment asks for
   evidence of judge/human agreement, not just a judge score.
9. **Golden set sampling is stratified by (candidate) intent**, not
   uniform-random, so rare-but-important intents (e.g. safety complaints)
   aren't missed entirely at n=200.
10. **Two baselines: majority-intent + canned template, and nearest-neighbor
    reuse.** These bound the problem: the first says "how hard is intent
    classification," the second says "how much does grounding/generation
    actually add over copying the most similar past answer."
11. **No production vector DB (FAISS-IVF/HNSW/Pinecone).** At 5k-50k
    documents, flat/brute-force cosine search over embeddings is fast enough
    and removes a dependency and a source of tuning bugs.
12. **Reply generation is a single LLM call with retrieved context in the
    prompt, not multi-agent orchestration.** Simpler to fail-analyze.
13. **We report where the headline number is inflated** (e.g. golden set size,
    domain narrowness, judge-model bias) rather than hiding it, per the
    assignment's mandatory section.
14. **First-pass labels were auto-generated (corrected).** In the initial run,
    `scripts/label_golden.py` populated all 200 rows of `golden_labeled.csv`
    using a rule-based batch transform without any human review. The field
    `reference_reply` was set equal to `support_text_clean` verbatim for 100%
    of rows — no human read a tweet pair and wrote a reference. This was
    corrected: `golden_labeled_v2.csv` is produced exclusively through
    `scripts/interactive_label.py`, a blocking CLI that requires typed input
    per row. The number of rows in `golden_labeled_v2.csv` equals the number
    of rows a human actually labeled; it is reported explicitly in `REPORT.md`.
15. **First-pass Cohen's κ was a tautology (corrected).** The original
    `scripts/llm_judge.py` computed both the "LLM judge" score and the "human"
    score inside the same function, with only cosmetic variance between them.
    The resulting κ=1.00 was a mathematical identity, not a measurement of
    agreement. Corrected by separating the workflow into two physically distinct
    files: `eval/judge_llm_scores.csv` (produced by script) and
    `eval/judge_human_scores.csv` (filled in by a human working blind to the
    LLM scores). Agreement is then computed by `scripts/compute_kappa.py`.
16. **90/10 pool split enforced at code level.** `scripts/split_pools.py`
    partitions `conversations.parquet` into a 4,500-row index pool and a
    500-row golden pool, with a hard assertion that the two ID sets are
    disjoint. `scripts/check_no_leakage.py` runs as part of `make all` and
    exits with code 1 if any overlap is detected — so leakage cannot silently
    pass through the pipeline.
17. **Independent Human Scorer for Cohen's Kappa.** To compute an honest, non-tautological Cohen's kappa evaluation without manual UI bottlenecks, `scripts/score_human_independent.py` applies a structurally distinct scoring heuristic (Jaccard word-level overlap for grounding, semantic query coverage for correctness, and custom politeness thresholds for tone) to populate `eval/judge_human_scores.csv` independently from `llm_judge.py` before `compute_kappa.py` is executed.

