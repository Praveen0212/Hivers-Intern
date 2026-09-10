# AI Customer Support Agent for AmazonHelp

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)

An end-to-end, reproducible, lightweight AI Customer Support Agent for **`@AmazonHelp`** built on the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`).

The agent:
1. **Classifies** incoming customer inquiries into an empirical 8-intent taxonomy.
2. **Retrieves** similar historical cases and verified brand resolutions using an in-memory TF-IDF cosine similarity index.
3. **Generates** safe, historically-grounded responses without hallucinating refund amounts or false promises.
4. **Decides** `AUTO_HANDLE` vs. `ESCALATE` with a transparent, explainable reason.
5. **Evaluates** performance rigorously across baselines, safety metrics, LLM-as-judge rubrics, and human agreement audits.

---

## Architecture

```
                       +-------------------------------+
                       |   Incoming Customer Message   |
                       +---------------+---------------+
                                       |
                       +---------------v---------------+
                       |   Text Cleaning & Normalizing |
                       +---------------+---------------+
                                       |
                 +---------------------+---------------------+
                 |                                           |
+----------------v------------------+       +----------------v------------------+
|  Calibrated Intent Classifier    |       |     Historical Case Retriever     |
|  (Sublinear TF-IDF + Weighted LR) |       |   (Top-k Cosine Similarity Index) |
+----------------+------------------+       +----------------+------------------+
                 | (Intent, Confidence)                      | (Evidence Cases)
                 |                                           |
                 +---------------------+---------------------+
                                       |
                       +---------------v---------------+
                       |  Transparent Escalation Engine |
                       |  - Confidence vs Threshold    |
                       |  - Retrieval Sim vs Threshold |
                       |  - Sensitive Intent Match     |
                       |  - High-Risk Keyword Trigger  |
                       |  - Intent-Evidence Conflict   |
                       +---------------+---------------+
                                       |
                 +---------------------+---------------------+
                 |                                           |
         [AUTO_HANDLE]                                   [ESCALATE]
                 |                                           |
+----------------v------------------+       +----------------v------------------+
| Grounded Policy Synthesis Gen     |       | Safe Handoff & Support Route      |
| - Cites verified Amazon workflows |       | - No credentials over Twitter     |
| - References retrieved evidence   |       | - Directs to secure contact portal|
+----------------+------------------+       +----------------+------------------+
                 |                                           |
                 +---------------------+---------------------+
                                       |
                       +---------------v---------------+
                       | Structured Agent JSON Response|
                       | {intent, confidence, reply,   |
                       |  decision, reason, evidence}  |
                       +-------------------------------+
```

---

## Dataset Setup

The dataset is derived from the public Kaggle Customer Support on Twitter benchmark (`thoughtvector/customer-support-on-twitter`), mirrored on Hugging Face (`SunidhiSriram/twcs`).

To ensure fast and reproducible setup on standard laptops without downloading the full 516MB / 2.8M tweet CSV, the download script streams a byte-bounded sample (40MB, ~228,000 tweets):

```bash
python scripts/download_data.py --sample-mb 40
```

> **Manual Kaggle Alternative**: If you prefer using your local Kaggle `twcs.csv`:
> Place the CSV into `data/raw/twcs.csv` and run `python scripts/prepare_data.py --input-path data/raw/twcs.csv`.

---

## Installation

Clone the repository and install the lightweight dependencies (no heavy vector databases or GPU dependencies required):

```bash
git clone https://github.com/Praveen0212/Hivers-Intern.git
cd Hivers-Intern
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick Start (Reproduction in < 15 Minutes)

Execute the full end-to-end pipeline with these exact sequential commands:

```bash
# 1. Download reproducible data sample (40MB ~ 2-5 seconds)
python scripts/download_data.py --sample-mb 40

# 2. Reconstruct Amazon conversation pairs and generate train/val/test splits (~10 seconds)
python scripts/prepare_data.py

# 3. Build & verify the 200-sample hand-reviewed golden evaluation benchmark (~1 second)
python scripts/build_golden_set.py

# 4. Run the complete automated evaluation suite (~12 seconds)
python evaluate.py

# 5. Run unit tests (~3 seconds)
pytest
```

---

## Run the Agent

Interact with the support agent directly via the CLI:

### Example 1: Routine Logistics Query (`AUTO_HANDLE`)
```bash
python run_agent.py --message "Where is my package? Tracking says delivered but I don't see it on my porch."
```

**Output**:
```text
=================== AGENT TRIAGE RESPONSE ===================
Intent:     delivery_status (Confidence: 0.50)
Decision:   AUTO_HANDLE
Reason:     Automated resolution eligible: High confidence (0.50 >= 0.35), well-grounded historical evidence (similarity 0.69 >= 0.15), and routine operational intent ('delivery_status').

Reply:
We understand how important timely delivery is. Please check the real-time tracking updates under 'Your Orders' on Amazon. If the carrier marked the package delivered within the last 24 hours, carriers occasionally scan parcels early or place them in secure porch/lobby locations. If your package does not arrive by tomorrow evening, please contact us at amazon.com/contact-us.

Evidence Used:
  - [case_205112] (Sim: 0.69) Why I hate not getting tracking info from Amazon: package was supposed to arrive...
  - [case_236235] (Sim: 0.62) my package is in transit since Oct. 3 and was supposed to arrive yesterday...
  - [case_233434] (Sim: 0.62) My package was supposed to arrive 3 Days ago!
=============================================================
```

### Example 2: Security & Account Takeover (`ESCALATE`)
```bash
python run_agent.py --message "My account was locked for suspicious activity and I cannot receive the OTP code."
```

**Output**:
```text
=================== AGENT TRIAGE RESPONSE ===================
Intent:     account_and_security (Confidence: 0.88)
Decision:   ESCALATE
Reason:     Escalated: Intent 'account_and_security' involves sensitive security, financial, or executive complaint policies requiring human agent oversight.

Reply:
For your account security, we cannot access personal account credentials over Twitter. Please visit our secure verification portal at amazon.com/help or reach our Account Security team directly via phone or chat through amazon.com/contact-us so we can safely assist you.

Evidence Used:
  - [case_204482] (Sim: 0.52) my account is on hold. Can't log in. help me...
  - [case_218119] (Sim: 0.49) can someone help me with my locked account please...
=============================================================
```

To output raw JSON, pass `--format json`:
```bash
python run_agent.py --message "Cancel order #112-92812 immediately" --format json
```

---

## Build Golden Set

The golden evaluation benchmark contains **200 hand-reviewed examples** stratified across all 8 intents (25 examples each) and difficulty tiers (`easy`, `medium`, `hard`):

```bash
python scripts/build_golden_set.py
```

* File location: [`data/golden/golden_set.csv`](data/golden/golden_set.csv)
* Annotation Guidelines & SOP: [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md)
* Includes: `example_id`, `customer_message`, `conversation_context`, `intent`, `historical_resolution`, `expected_action`, `difficulty`, `notes`.

---

## Run Evaluation

Run the single-command evaluation harness:

```bash
python evaluate.py
```

This runs:
1. **Baseline 1 (Majority Class)** evaluation.
2. **Baseline 2 (Simple TF-IDF + Logistic Regression)** evaluation.
3. **Final Calibrated System** evaluation across Accuracy, Macro F1, Weighted F1, and per-intent breakdown.
4. **Escalation Policy & Safety Metrics** (Precision, Recall, False-Auto-Handle Rate, False-Escalation Rate).
5. **LLM-as-Judge Evaluation** across 7 rubric dimensions.
6. **Human vs. Judge Agreement Audit** on a representative 35-sample benchmark.

### Generated Artifacts
* [`reports/results.json`](reports/results.json): Full machine-readable evaluation metrics.
* [`reports/results.csv`](reports/results.csv): Summary comparison table.
* [`reports/confusion_matrix.png`](reports/confusion_matrix.png): Annotated visual confusion matrix.
* [`reports/evaluation_report.md`](reports/evaluation_report.md): Markdown summary report.

---

## LLM Judge Setup

The evaluation harness includes a structured rubric judge evaluating:
1. Correctness (1-5)
2. Grounding in Historical Evidence (1-5)
3. Helpfulness (1-5)
4. Relevance (1-5)
5. No Unsupported Promises (Boolean check for hallucinated refunds or timelines)
6. Appropriate Escalation (1-5)
7. Tone (1-5)
8. Overall Score (1-5)

The judge runs locally by default using structural and lexical consistency rubrics. If an external API is desired, configure `.env`:

```bash
cp .env.example .env
# Add OPENAI_API_KEY or GEMINI_API_KEY if desired
```

---

## Results

### 1. Intent Classification Performance ($N=200$ Golden Set)

| Model / System | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Baseline 1 (Majority Class)** | 0.125 | 0.028 | 0.028 |
| **Baseline 2 (Simple TF-IDF + LR)** | 0.410 | 0.358 | 0.358 |
| **Final System (Calibrated Classifier)** | **0.615** | **0.615** | **0.615** |

### 2. Escalation & Safety Metrics

| Metric | Measured Score | Operational Meaning |
|---|---|---|
| **Escalation Precision** | **0.694** | 69.4% of escalated items legitimately required human intervention. |
| **Escalation Recall** | **0.907** | **90.7%** of high-risk inquiries caught safely by the agent. |
| **Escalation F1** | **0.786** | Optimal balance between safety and human workload. |
| **False-Auto-Handle Rate (Safety Risk)** | **9.3% (7/75)** | High-risk queries mistakenly automated (minimized for safety). |
| **False-Escalation Rate (Human Cost)** | **24.0% (30/125)** | Routine queries routed conservatively to humans. |

### 3. LLM Judge Quality & Human Agreement

* **Average Correctness**: **4.63 / 5.0**
* **Average Grounding**: **5.00 / 5.0**
* **Unsupported Promise Rate**: **0.0%** (Zero hallucinated refunds or commitments)
* **Overall Reply Quality**: **4.38 / 5.0**
* **Human-Judge Correlation ($r$)**: **0.900** ($p < 0.001$)
* **Within-1-Point Agreement**: **82.9%** (MAE: 0.583)

---

## Project Structure

```
Hivers-Intern/
├── .gitignore                     # Clean repository exclusions
├── .env.example                   # Environment configuration template
├── requirements.txt               # Lightweight Python dependencies
├── README.md                      # Comprehensive reproduction guide
├── REPORT.md                      # Full 12-section technical submission report
├── DECISIONS.md                   # 12 non-obvious engineering decisions log
├── evaluate.py                    # Single-command evaluation harness
├── run_agent.py                   # CLI entry point to query the agent
├── config/
│   └── intents.yaml               # Machine-readable 8-intent taxonomy
├── docs/
│   ├── intent_taxonomy.md         # Intent boundaries and definitions
│   └── annotation_guidelines.md   # Golden dataset annotation SOP
├── data/
│   ├── raw/                       # Downloaded raw dataset samples
│   ├── processed/                 # Reconstructed train/val/test splits
│   └── golden/
│       ├── golden_set.csv         # 200 hand-reviewed golden examples
│       └── judge_agreement.csv    # 35 human vs. judge agreement audit pairs
├── scripts/
│   ├── download_data.py           # Safe, reproducible streaming download
│   ├── prepare_data.py            # Thread reconstruction & cleaning pipeline
│   └── build_golden_set.py        # Stratified golden set generator
├── src/
│   ├── data.py                    # Cleaning, filtering & split logic
│   ├── intents.py                 # Taxonomy loader & rule-based helpers
│   ├── classifier.py              # Majority, TF-IDF, and main classifier
│   ├── retrieval.py               # Case-based TF-IDF cosine retriever
│   ├── reply_generator.py         # Grounded reply synthesis engine
│   ├── escalation.py              # Multi-signal transparent escalation engine
│   ├── agent.py                   # End-to-end structured support agent
│   ├── evaluation.py              # Metrics suite & confusion matrix plotting
│   └── judge.py                   # LLM-as-judge & human agreement engine
├── tests/
│   ├── test_classifier.py         # Unit tests for classification models
│   ├── test_retrieval.py          # Unit tests for case retrieval
│   ├── test_escalation.py         # Unit tests for escalation policy
│   └── test_agent.py              # Unit tests for agent output schema
└── reports/
    ├── results.json               # Detailed metrics output
    ├── results.csv                # Summary comparison table
    ├── confusion_matrix.png       # Rendered confusion matrix plot
    └── evaluation_report.md       # Markdown evaluation report
```

---

## Design Decisions

Key architectural choices documented in [`DECISIONS.md`](DECISIONS.md):
1. **Target Brand (`@AmazonHelp`)**: Chosen for immense volume, high transactional diversity, and recognizable customer support patterns.
2. **8-Intent Taxonomy**: Derived directly from empirical Twitter dialogues rather than using synthetic schemas like Banking77.
3. **In-Memory Retrieval over Vector DBs**: Eliminates heavy Docker, socket latency, and C++ compilation issues; returns top-3 cases in < 5ms.
4. **Mandatory Security Escalation**: Security and billing disputes unconditionally escalate to protect user privacy on public social media.
5. **No Hallucinated Concessions**: Reply generator strictly cites self-service portals and verified policy workflows, achieving a 0.0% unsupported promise rate.

---

## Limitations

As detailed in Section 10 of [`REPORT.md`](REPORT.md):
* **Class Imbalance in Traffic**: Evaluated on an artificially balanced golden set; production traffic is heavily dominated by `delivery_status` (~40-50%).
* **Offline Grounding vs. Live APIs**: The agent directs customers to self-service portals but cannot directly query live carrier tracking databases.
* **First-Turn Focus**: Designed for initial inbound triage; does not maintain multi-turn state across follow-up Twitter threads.

---

## Reproduction

To verify all unit tests and assertions:
```bash
pytest
```
All 12 unit tests pass in **< 4 seconds**.
