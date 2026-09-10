# Automated Evaluation Report

## 1. Intent Classification Performance

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Baseline 1 (Majority Class)** | 0.125 | 0.028 | 0.028 |
| **Baseline 2 (Simple TF-IDF + LR)** | 0.410 | 0.358 | 0.358 |
| **Final System (Calibrated Classifier)** | **0.615** | **0.615** | **0.615** |

### Per-Intent Performance (Final System)

| Intent | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `account_and_security` | 1.000 | 0.440 | 0.611 | 25 |
| `damaged_or_missing` | 0.842 | 0.640 | 0.727 | 25 |
| `delivery_status` | 0.338 | 0.960 | 0.500 | 25 |
| `general_feedback_or_complaint` | 0.591 | 0.520 | 0.553 | 25 |
| `order_cancellation_or_change` | 1.000 | 0.280 | 0.438 | 25 |
| `payment_and_billing` | 0.600 | 0.360 | 0.450 | 25 |
| `return_and_refund` | 0.719 | 0.920 | 0.807 | 25 |
| `subscription_and_prime` | 0.870 | 0.800 | 0.833 | 25 |

---

## 2. Escalation & Safety Metrics

The escalation arbiter decides whether to safely automate or escalate to human specialists:

* **Escalation Precision**: `0.694`
* **Escalation Recall**: `0.907`
* **Escalation F1-Score**: `0.786`
* **False-Auto-Handle Rate (High Safety Risk)**: `9.3%` (7/75)
* **False-Escalation Rate (Human Burden Cost)**: `24.0%` (30/125)

---

## 3. LLM-as-Judge Reply Quality (1-5 Scale)

* **Correctness**: `4.63/5.0`
* **Grounding in Historical Evidence**: `5.00/5.0`
* **Helpfulness**: `3.94/5.0`
* **Relevance**: `3.38/5.0`
* **Unsupported Claims / Hallucination Rate**: `0.0%`
* **Overall Reply Quality**: `4.38/5.0`

---

## 4. Human vs. LLM Judge Agreement Validation (n=35)

* **Mean Absolute Error (MAE)**: `0.583`
* **Pearson Correlation (r)**: `0.900`
* **Spearman Rank Correlation (rho)**: `0.432`
* **Within-1-Point Agreement Rate**: `82.9%`
