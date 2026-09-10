# Golden Dataset Annotation Guidelines

## Overview
The golden evaluation set provides a ground-truth benchmark of 200 customer support interactions for `@AmazonHelp`. This document establishes the standard operating procedure (SOP) for labeling, adjudicating edge cases, and assigning expected agent actions.

---

## Dataset Schema

Each entry in `data/golden/golden_set.csv` must include the following required fields:

| Field | Description | Type | Examples |
|---|---|---|---|
| `example_id` | Unique identifier | String | `gold_001`, `gold_045` |
| `customer_message` | Raw customer text received on Twitter | String | *"Where is my package? Supposed to arrive yesterday"* |
| `conversation_context` | Prior context or channel details | String | `Twitter direct mention (@AmazonHelp)` |
| `intent` | Gold intent label (must match taxonomy) | String | `delivery_status`, `account_and_security` |
| `historical_resolution`| Verified historical resolution or guideline | String | *"Direct to Your Orders tracking; ask for DM if overdue"* |
| `expected_action` | Correct triage decision | String | `AUTO_HANDLE` or `ESCALATE` |
| `difficulty` | Complexity rating for evaluation stratification | String | `easy`, `medium`, `hard` |
| `notes` | Human annotator justification / boundary notes | String | *"Mentions late delivery + angry tone; primary intent is delivery"* |

---

## Difficulty Classification Criteria

1. **Easy (`easy`)**:
   - Contains explicit, unambiguous keywords (e.g., *"tracking number says delivered but no package"*, *"I want to cancel order"*).
   - Single intent, standard grammar and vocabulary.

2. **Medium (`medium`)**:
   - Implied intent or conversational phrasing without exact keywords (e.g., *"Still waiting by the door all evening..."*).
   - Mild emotional charge or multiple sentence descriptions.

3. **Hard (`hard`)**:
   - Competing intents in one message (e.g., *"The driver threw my package, broke the plate, and now I want a refund"* -> Primary: `damaged_or_missing`).
   - Sarcasm, abbreviations, typos, or indirect demands (e.g., *"Another day, another phantom delivery from Amazon"*).
   - Edge cases testing strict escalation boundaries (e.g., sensitive account lockout vs routine login help).

---

## Escalation Decision Rules

Assign `ESCALATE` if any of the following apply:
1. **Security & Authentication**: Any mention of hacked accounts, unauthorized purchases, 2FA/OTP failures, or account closures (`account_and_security`).
2. **Financial & Billing Disputes**: Direct claims of duplicate credit card charges or unapplied promotional codes requiring financial verification (`payment_and_billing`).
3. **Severe Complaints / Demands for Supervisors**: Demands to speak with a manager, threats of legal action, or driver misconduct (`general_feedback_or_complaint`).
4. **Ambiguous or Insufficient Information**: Inquiries where the customer's request cannot be addressed safely without private customer verification.

Assign `AUTO_HANDLE` if:
1. The request is routine, self-serviceable via public Amazon portal tools (*Your Orders*, *Online Returns Center*, *Prime Settings*), or standard policy information (e.g., typical 3–5 day refund turnaround).
