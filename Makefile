.PHONY: all split index golden run evaluate judge clean

# ─── Full pipeline (reproducible) ────────────────────────────────────────────
all: split index discover golden

split:
	python scripts/prepare_data.py --input data/raw/twcs.csv --brand AmericanAir \
		--sample_conversations 5000 --seed 42 --output data/processed/conversations.parquet
	python scripts/split_pools.py \
		--input data/processed/conversations.parquet \
		--output_dir data/processed --seed 42
	python scripts/check_no_leakage.py \
		--index_pool data/processed/conversations_index_pool.parquet \
		--golden_pool data/processed/conversations_golden_pool.parquet

index:
	python scripts/build_index.py \
		--conversations data/processed/conversations_index_pool.parquet \
		--output data/processed/reply_index

discover:
	python scripts/discover_intents.py \
		--conversations data/processed/conversations_golden_pool.parquet \
		--n_clusters 8 --seed 42 \
		--output data/processed/intent_taxonomy_golden.json
	python scripts/sample_golden.py \
		--conversations data/processed/conversations_golden_pool.parquet \
		--taxonomy data/processed/intent_taxonomy_golden.json \
		--n 200 --seed 42 \
		--output golden/golden_unlabeled_v2.csv
	@echo ""
	@echo "============================================================"
	@echo "  NEXT STEP (HUMAN REQUIRED)"
	@echo "============================================================"
	@echo "  Run the interactive labeling CLI:"
	@echo "    python scripts/interactive_label.py"
	@echo "  Then run 'make run' when you have labeled at least 60 rows."
	@echo "============================================================"

# ─── After human labeling is done ────────────────────────────────────────────
run:
	python scripts/run_agent.py \
		--golden golden/golden_labeled_v2.csv \
		--index data/processed/reply_index \
		--output eval/predictions_v2.csv
	python scripts/check_no_leakage.py \
		--index_pool data/processed/conversations_index_pool.parquet \
		--golden_pool data/processed/conversations_golden_pool.parquet \
		--predictions eval/predictions_v2.csv

evaluate:
	python scripts/evaluate.py \
		--predictions eval/predictions_v2.csv \
		--golden golden/golden_labeled_v2.csv \
		--output eval/results_v2.json

judge:
	python scripts/llm_judge.py \
		--predictions eval/predictions_v2.csv \
		--n_samples 30 --seed 42
	@echo ""
	@echo "============================================================"
	@echo "  NEXT STEP (HUMAN REQUIRED)"
	@echo "============================================================"
	@echo "  Open eval/judge_human_blank.csv in Excel/Sheets."
	@echo "  Read prompts/llm_judge_rubric.txt for the rubric."
	@echo "  Score grounding/correctness/tone (0, 2, or 4) per row."
	@echo "  Do NOT look at eval/judge_llm_scores.csv while scoring."
	@echo "  Save as eval/judge_human_scores.csv, then run:"
	@echo "    python scripts/compute_kappa.py"
	@echo "============================================================"

clean:
	rm -rf data/processed/* eval/*.csv eval/*.json


