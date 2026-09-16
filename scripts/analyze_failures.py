import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

preds = pd.read_csv('eval/predictions.csv')
gold = pd.read_csv('golden/golden_labeled.csv')
merged = preds.merge(gold, on='customer_tweet_id')

errors = merged[merged['predicted_intent'] != merged['intent_label']]
print(f"Total intent errors: {len(errors)} / {len(merged)}")
print("\nSample intent errors:")
count = 0
for i, r in errors.iterrows():
    print(f"Index: {i}")
    print(f"Query: {r['customer_text_clean_x']}")
    print(f"Gold Intent: {r['intent_label']} vs Pred Intent: {r['predicted_intent']}")
    print(f"Agent Reply: {r['generated_reply']}")
    print("-" * 50)
    count += 1
    if count >= 8:
        break

esc_errors = merged[(merged['escalate_x'].astype(str).str.lower() == 'true') != (merged['escalate_y'].astype(str).str.lower() == 'yes')]
print(f"\nTotal escalation errors: {len(esc_errors)}")
for i, r in esc_errors.iterrows():
    print(f"Query: {r['customer_text_clean_x']}")
    print(f"Gold Escalate: {r['escalate_y']} vs Pred Escalate: {r['escalate_x']}")
    print(f"Gold Reason: {r['escalate_reason_y']}")
    print(f"Pred Reason: {r['escalate_reason_x']}")
    print("-" * 50)
