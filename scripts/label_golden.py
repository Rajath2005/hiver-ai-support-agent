"""
Labeling script for golden dataset.
Assigns accurate intent labels, escalation flags, escalation reasons, and reference replies.
"""
import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('golden/golden_unlabeled.csv')

def classify_row(c_text, s_text, suggested):
    c_lower = c_text.lower()
    
    # Intent classification
    intent = suggested
    
    # Baggage
    if any(w in c_lower for w in ['bag', 'bags', 'baggage', 'luggage', 'suitcase', 'carousel']):
        intent = "BaggageIssue"
    # Seating & Boarding
    elif any(w in c_lower for w in ['seat', 'seats', 'seating', 'bulkhead', 'upgrade', 'boarding', 'board', 'first class', 'exit row']):
        intent = "SeatingAndBoarding"
    # Delay & Disruptions
    elif any(w in c_lower for w in ['delay', 'delayed', 'delays', 'late', 'hours late', 'cancelled', 'cancellation', 'missed connection', 'diverted', 'stuck on tarmac']):
        intent = "FlightDelaysAndDisruptions"
    # Ticketing & Refunds
    elif any(w in c_lower for w in ['refund', 'voucher', 'ticket', 'fare', 'price', 'book a flight', 'book flight', 'change fee', 'cost', 'charge $', 'receipt', 'miles', 'advantage miles']):
        intent = "TicketingAndRefunds"
    # Direct Assistance / DM
    elif any(w in c_lower for w in ['phone number', 'call center', 'dm', 'direct message', 'sent dm', 'record locator', 'booking reference', 'contact number']):
        intent = "DirectAssistanceOrDM"
    # Compliments & Appreciation
    elif any(w in c_lower for w in ['thank you', 'thanks', 'great job', 'kudos', 'wonderful', 'amazing', 'love flying', 'shout out', 'cookies', 'best crew']):
        intent = "ComplimentOrAppreciation"
    # Customer Service Complaint
    elif any(w in c_lower for w in ['rude', 'attitude', 'terrible service', 'worst customer service', 'shame on you', 'disrespectful', 'unhelpful']):
        intent = "CustomerServiceComplaint"
    # General Flight Feedback fallback
    elif suggested == "GeneralFlightFeedback":
        intent = "GeneralFlightFeedback"
    
    # Escalation determination
    escalate = "no"
    reason = "Standard informational inquiry addressable by automated guidance and policy links."
    
    # Safety / Medical / Emergency
    if any(w in c_lower for w in ['medical', 'emergency', 'hurt', 'injury', 'injured', 'sick', 'unsafe', 'danger', 'assault', 'threat']):
        escalate = "yes"
        reason = "Medical, injury, or safety-related incident requiring immediate human intervention."
    # Legal / Regulatory
    elif any(w in c_lower for w in ['lawyer', 'attorney', 'lawsuit', 'sue', 'legal', 'dot complaint', 'faa']):
        escalate = "yes"
        reason = "Legal threat or regulatory reporting requiring human legal/customer relations escalation."
    # Explicit human request
    elif any(w in c_lower for w in ['manager', 'supervisor', 'human', 'representative', 'real person', 'speak to someone']):
        escalate = "yes"
        reason = "Customer explicitly requested a human representative or supervisor."
    # Severe disruption / Stranded / High distress / Financial dispute
    elif any(w in c_lower for w in ['stranded', 'never fly aa', 'never again', 'stolen', 'ruined our trip', 'ruined vacation', 'scam', 'fraud', 'missing funeral']):
        escalate = "yes"
        reason = "Severe distress, high-impact disruption, or severe financial/property dispute requiring human empathy and discretion."
    elif any(w in c_lower for w in ['lost my bag', 'lost bag', 'lost luggage', 'damaged bag']):
        # If lost baggage claim
        escalate = "yes"
        reason = "Lost or damaged baggage report requiring personalized claim tracing."

    ref_reply = s_text.strip()
    return intent, escalate, reason, ref_reply

labeled_rows = []
for idx, row in df.iterrows():
    c_text = str(row['customer_text_clean'])
    s_text = str(row['support_text_clean'])
    sugg = str(row['suggested_intent'])
    
    intent, escalate, reason, ref_reply = classify_row(c_text, s_text, sugg)
    
    labeled_rows.append({
        'customer_tweet_id': row['customer_tweet_id'],
        'support_tweet_id': row['support_tweet_id'],
        'customer_text_clean': c_text,
        'support_text_clean': s_text,
        'suggested_intent': sugg,
        'intent_label': intent,
        'escalate': escalate,
        'escalate_reason': reason,
        'reference_reply': ref_reply
    })

res_df = pd.DataFrame(labeled_rows)
res_df.to_csv('golden/golden_labeled.csv', index=False, encoding='utf-8')
print(f"Labeled {len(res_df)} rows and saved to golden/golden_labeled.csv")
print("\nIntent Distribution:")
print(res_df['intent_label'].value_counts())
print("\nEscalation Distribution:")
print(res_df['escalate'].value_counts())
