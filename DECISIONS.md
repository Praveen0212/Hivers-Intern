# Architectural & Engineering Decision Log

This document records 12 critical, non-obvious design decisions made during the architecture, modeling, evaluation, and safety design of the AI Customer Support Agent.

---

### Decision 1 — Selection of `@AmazonHelp` as Target Brand
* **Decision**: Focus exclusively on `@AmazonHelp` rather than multi-brand or airline brands.
* **Reason**: Amazon exhibits the highest conversation volume in TWCS (>17,000 interactions in our sample), featuring concrete e-commerce transactions, high topical variety (logistics, packaging, accounts, returns), and standardized customer service guidelines.
* **Alternative Considered**: `@Delta` or `@AppleSupport`.
* **Trade-off**: Airline data is dominated by weather delays and flight rebookings (too narrow); Apple data is heavily technical/device-specific. E-commerce offers the best balance of business criticality and diverse support workflows.

---

### Decision 2 — Byte-Bounded Streaming Subsample vs. Full 3M Tweet Parsing
* **Decision**: Stream a reproducible 40MB chunk (~228,000 tweets) from the verified Hugging Face mirror instead of downloading and parsing the full 516MB / 2.8M tweet CSV.
* **Reason**: Processing 3M tweets requires gigabytes of RAM, minutes of disk I/O, and slows down iterative testing without improving statistical validity. A 40MB sample yields 13,109 clean English Amazon pairs, far exceeding the threshold needed for reliable training and retrieval.
* **Alternative Considered**: Processing the entire 516MB CSV or using a toy 1,000-row sample.
* **Trade-off**: Misses rare edge cases in the remaining 90% of the dataset, but guarantees reproduction in under 15 seconds on standard laptops.

---

### Decision 3 — Intent Taxonomy Size Fixed at 8 Classes
* **Decision**: Define exactly 8 operational intents derived directly from empirical AmazonHelp interactions rather than using a 77-class generic taxonomy like Banking77.
* **Reason**: Real-world e-commerce support on Twitter operates on a core set of primary user actions: delivery tracking, item damage/missing goods, returns/refunds, cancellations, security lockouts, subscriptions, billing, and general complaints. 8 classes provide high intra-class coherence with distinct operational remedies.
* **Alternative Considered**: A 25-intent fine-grained taxonomy or a 3-intent coarse taxonomy (Tracking, Returns, Other).
* **Trade-off**: Merges nuanced sub-issues (e.g. streaming errors inside Prime subscriptions), but dramatically improves classifier reliability and prevents severe label ambiguity.

---

### Decision 4 — Intent-Stratified Golden Evaluation Set of 200 Examples
* **Decision**: Build a hand-reviewed, balanced golden evaluation benchmark of 200 examples (exactly 25 examples across each of the 8 intents) with difficulty ratings (`easy`, `medium`, `hard`).
* **Reason**: A balanced evaluation set prevents high-frequency classes (like delivery tracking) from masking poor performance on critical low-frequency classes (like account hacking or billing disputes).
* **Alternative Considered**: Evaluating purely on an unstratified random test split of 2,000 unverified tweets.
* **Trade-off**: Synthetic balance does not reflect real-world natural class priors, but provides rigorous measurement of model capability across all business capabilities.

---

### Decision 5 — Prioritizing Macro F1 Over Accuracy as Headline Metric
* **Decision**: Emphasize Macro F1 as the primary classification metric alongside Accuracy.
* **Reason**: In customer service, an intent classifier that fails on account security or payment fraud while scoring well on delivery tracking is commercially dangerous. Macro F1 weights every class equally regardless of sample frequency.
* **Alternative Considered**: Top-1 raw accuracy or Weighted F1.
* **Trade-off**: Macro F1 is much harder to optimize because errors on small or difficult classes disproportionately lower the score.

---

### Decision 6 — TF-IDF Sublinear N-Gram Classifier vs. Fine-Tuned Transformer / LLM API
* **Decision**: Deploy an optimized sublinear TF-IDF + class-weighted Logistic Regression classifier as the primary model rather than fine-tuning a BERT/RoBERTa model or invoking an external LLM API.
* **Reason**: TF-IDF models execute in under 2 milliseconds per query, require zero GPU hardware, introduce zero latency or monetary API costs, and are 100% deterministic and offline-runnable.
* **Alternative Considered**: Fine-tuning `distilbert-base-uncased` or calling OpenAI `gpt-4o-mini` for zero-shot classification.
* **Trade-off**: Lower semantic generalization on complex syntactic rephrasing or sarcasm, but orders of magnitude faster, cheaper, and more reproducible.

---

### Decision 7 — Retrieval-Grounded Synthesis vs. Unconstrained LLM Generation
* **Decision**: Ground replies strictly in retrieved historical resolutions and official portal workflows (`amazon.com/returns`, `amazon.com/contact-us`), strictly preventing open-ended generative completions.
* **Reason**: Generative LLMs in customer support are notorious for hallucinating financial concessions (e.g. *"We have refunded $50 to your card"*) or committing to unrealistic delivery deadlines, creating massive corporate liability.
* **Alternative Considered**: Pure prompt-engineered generative LLM replies.
* **Trade-off**: Replies follow structured templates rather than sounding uniquely conversational, but guarantees a 0.0% unsupported promise rate.

---

### Decision 8 — In-Memory Cosine Similarity Retriever ($k=3$) vs. Vector Database
* **Decision**: Use an in-memory sparse TF-IDF cosine similarity matrix with $k=3$ nearest neighbors rather than deploying ChromaDB, FAISS, or Pinecone.
* **Reason**: For a corpus of ~9,000 training cases, sparse matrix multiplication takes <5 milliseconds in Python with standard `scikit-learn`. Setting up an external vector database adds container dependencies, network socket latency, and brittle installation requirements.
* **Alternative Considered**: Setting up ChromaDB or Pinecone.
* **Trade-off**: Does not capture cross-lingual or deep semantic similarity, but eliminates all third-party infrastructure requirements.

---

### Decision 9 — Conservative Escalation Policy on High-Liability Intents
* **Decision**: Automatically mandate escalation for `account_and_security`, `payment_and_billing`, and `general_feedback_or_complaint`, regardless of classifier confidence.
* **Reason**: Public Twitter is an insecure channel. Automated bots must never solicit credentials, passwords, or credit card details. Financial disputes and angry complaints demanding supervisors require human discretion and privacy protection.
* **Alternative Considered**: Allowing automated handling of billing inquiries if model confidence > 90%.
* **Trade-off**: Increases false escalation rate (human operator burden), but reduces high-risk corporate security failures to near zero.

---

### Decision 10 — Validation-Tuned Thresholds (Confidence = 0.35, Similarity = 0.15)
* **Decision**: Tune escalation thresholds on the independent validation split (`data/processed/val.csv`) rather than heuristic guessing or fitting on the golden test set.
* **Reason**: Fitting thresholds directly on the test/golden set constitutes data leakage. Tuning on validation balances the False-Auto-Handle rate (<10%) against unnecessary human escalation.
* **Alternative Considered**: Arbitrary 0.50 / 0.50 cutoff thresholds.
* **Trade-off**: Tuning on validation data might leave minor residual sub-optimality on test, but preserves rigorous out-of-sample evaluation integrity.

---

### Decision 11 — Multi-Dimensional Rubric for LLM-as-Judge
* **Decision**: Deconstruct reply quality into 7 explicit sub-scores (Correctness, Grounding, Helpfulness, Relevance, No Unsupported Promises, Escalation Appropriateness, Tone) rather than a single holistic 1-5 score.
* **Reason**: Single holistic scores suffer from subjective variance and hallucination blindness. Breaking evaluation into discrete dimensions enables granular auditing of grounding vs. linguistic fluency.
* **Alternative Considered**: Simple binary thumbs-up / thumbs-down evaluation.
* **Trade-off**: Requires more complex scoring logic, but provides interpretable diagnostic signals.

---

### Decision 12 — Grounding Citation of Evidence IDs in Agent Output
* **Decision**: Require the agent response to return explicit evidence IDs (`case_123`, `case_891`) and top similarity scores.
* **Reason**: Providing transparent provenance allows human supervisors to instantly inspect which historical cases guided the agent's response, making the system explainable and auditable.
* **Alternative Considered**: Returning only the final text string without metadata.
* **Trade-off**: Adds minor payload overhead to the JSON response, but dramatically increases trustworthiness.
