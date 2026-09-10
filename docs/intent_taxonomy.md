# AmazonHelp Customer Support Intent Taxonomy

## 1. Intent Discovery Methodology

This intent taxonomy was empirically derived from analyzing over 17,000 real customer interactions directed at `@AmazonHelp` within the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`). 

Rather than adopting an arbitrary or synthetic classification scheme (such as Banking77), this taxonomy reflects the actual operational distribution of e-commerce inquiries handled on Twitter:
- High volume of logistical status checks (*"Where is my order?"*).
- Physical condition issues upon delivery (*damaged or missing items*).
- Post-purchase workflows (*returns, refunds, and cancellations*).
- Account security concerns requiring specialized escalation protocols.
- Digital and Prime subscription maintenance.

A target size of **8 distinct intents** was selected to ensure maximum operational utility, high intra-class coherence, and clear decision boundaries.

---

## 2. Intent Specifications & Decision Boundaries

### 1. `delivery_status`
* **Name**: Delivery & Tracking Status
* **Definition**: Customer inquiring about shipment progress, delayed packages, carrier hand-offs, tracking updates, or packages marked delivered but missing from doorstep.
* **Representative Inquiries**:
  - *"My order was supposed to arrive today by 8pm but the tracking hasn't updated since yesterday."*
  - *"It says my package was delivered to my porch, but I checked everywhere and there is nothing here."*
  - *"Where is my package? The tracking number says in transit for 5 days."*
* **Decision Boundaries**:
  - **vs. `damaged_or_missing`**: If the package *has* arrived and was opened, but an item inside is missing or broken, classify as `damaged_or_missing`. Use `delivery_status` only when the shipment or parcel itself is missing/in-transit.
  - **vs. `order_cancellation_or_change`**: If the customer states *"it's so late, cancel it now"*, prioritize `order_cancellation_or_change`.
* **Expected System Action**: `AUTO_HANDLE` (Routine informational and status inquiry).
* **Historical Resolution Pattern**: Prompt customer to check tracking via *Your Orders*, inspect immediate surroundings / safe drop locations, verify carrier delivery windows, and provide a secure link if the estimated window has passed.

---

### 2. `damaged_or_missing`
* **Name**: Damaged, Defective, or Missing Items
* **Definition**: The package arrived, but contents are broken, visibly defective, leaking, or specific purchased items were missing from the sealed box.
* **Representative Inquiries**:
  - *"My package arrived completely ripped open and 2 items are missing from inside."*
  - *"I ordered a coffee maker but received a box of phone cables instead."*
  - *"The glass bowl I ordered arrived shattered into pieces."*
* **Decision Boundaries**:
  - **vs. `return_and_refund`**: If an item is undamaged and the customer simply wants to return it for a refund, use `return_and_refund`.
  - **vs. `delivery_status`**: If the entire box never arrived, use `delivery_status`.
* **Expected System Action**: `AUTO_HANDLE` (Direct to self-service Online Return Center for instant replacement).
* **Historical Resolution Pattern**: Express sincere regret for condition upon arrival, route customer to Amazon's *Online Return Center* (`amazon.com/returns`) to request an immediate zero-cost replacement or return authorization.

---

### 3. `return_and_refund`
* **Name**: Returns & Refund Inquiries
* **Definition**: Inquiries regarding the return process, return labels, drop-off partners (UPS Store, Kohl's, Whole Foods), or tracking a pending refund credit.
* **Representative Inquiries**:
  - *"I returned my headphones at the UPS store 4 days ago. When will I see my refund?"*
  - *"Can I drop off my Amazon return at Whole Foods without printing a label?"*
  - *"I sent the item back two weeks ago and haven't received my refund."*
* **Decision Boundaries**:
  - **vs. `payment_and_billing`**: Inquiries specifically tied to the return of goods belong here. Unexplained bank charges or gift card issues belong in `payment_and_billing`.
  - **vs. `order_cancellation_or_change`**: Canceling an order before shipment belongs in `order_cancellation_or_change`.
* **Expected System Action**: `AUTO_HANDLE` (Clarify standard 3-5 business day refund window; explain QR code return drop-offs).
* **Historical Resolution Pattern**: Outline the standard refund timeline (typically 3–5 business days following warehouse inspection), explain QR-code label-free drop-off at approved partners, and link to *Your Orders* refund tracker.

---

### 4. `order_cancellation_or_change`
* **Name**: Order Cancellation & Address/Payment Modification
* **Definition**: Customer wanting to cancel an existing order before dispatch, modify the delivery address, or update payment information on an active order.
* **Representative Inquiries**:
  - *"I accidentally placed an order to my old apartment address, can I update it to my current home?"*
  - *"I want to cancel order #112-3849182 immediately before it ships."*
  - *"Can I cancel the preorder I made last week?"*
* **Decision Boundaries**:
  - **vs. `subscription_and_prime`**: Canceling Amazon Prime membership belongs in `subscription_and_prime`.
  - **vs. `account_and_security`**: Requesting total account closure belongs in `account_and_security`.
* **Expected System Action**: `AUTO_HANDLE` (Instruct immediate cancellation via self-service portal).
* **Historical Resolution Pattern**: Instruct customer to open *Your Orders* and select *Cancel Items* if still in processing; clarify that once dispatched, address cannot be changed and the package must be returned or refused upon delivery.

---

### 5. `account_and_security`
* **Name**: Account Access, Security & Authentication
* **Definition**: Issues with locked accounts, 2FA/OTP failures, password reset problems, suspected account takeover, or account closure requests.
* **Representative Inquiries**:
  - *"My account has been locked for suspicious activity and I cannot log in to access my orders."*
  - *"I am not receiving the OTP verification code on my phone to log in."*
  - *"Someone accessed my account and placed orders I didn't authorize. Help!"*
* **Decision Boundaries**:
  - **vs. `payment_and_billing`**: Routine card declines with active access belong in `payment_and_billing`. Security lockouts or fraudulent account takeovers belong here.
* **Expected System Action**: `ESCALATE` (High risk; automated agents must never collect credentials or handle compromised accounts).
* **Historical Resolution Pattern**: Emphasize security caution (never share credentials on public social media), direct to the verified two-step verification recovery page, and escalate to account specialists.

---

### 6. `subscription_and_prime`
* **Name**: Prime Membership & Digital Subscriptions
* **Definition**: Questions regarding Amazon Prime membership billing, auto-renewal cancellations, Prime Video/Music streaming errors, or student/household discounts.
* **Representative Inquiries**:
  - *"I was charged $139 for Prime renewal today, but I don't want it anymore. Can I cancel and get refunded?"*
  - *"Prime Video is giving me error code 5004 on my smart TV."*
  - *"How do I sign up for the student Prime discount?"*
* **Decision Boundaries**:
  - **vs. `delivery_status`**: Physical orders using Prime shipping belong in `delivery_status`.
  - **vs. `payment_and_billing`**: Subscriptions to Amazon Prime/Music belong here; generic credit card double-charges belong in `payment_and_billing`.
* **Expected System Action**: `AUTO_HANDLE` (Direct to Prime membership settings; provide device troubleshooting).
* **Historical Resolution Pattern**: Guide user to *Manage Prime Membership*, explain pro-rated refund availability if benefits were unused, and provide basic restart/reinstall steps for digital streaming.

---

### 7. `payment_and_billing`
* **Name**: Payment, Gift Cards & Billing Disputes
* **Definition**: Inquiries regarding duplicate card charges, gift card redemption failures, invalid promotional credits, or declined payment instruments.
* **Representative Inquiries**:
  - *"My credit card was charged twice for the same purchase."*
  - *"My $50 Amazon gift card says claim code is invalid or already redeemed."*
  - *"I applied a promo code at checkout but the discount was not reflected on my total."*
* **Decision Boundaries**:
  - **vs. `return_and_refund`**: Refunds stemming from returned goods belong in `return_and_refund`.
  - **vs. `account_and_security`**: Unauthorized charges linked to a stolen login belong in `account_and_security`.
* **Expected System Action**: `ESCALATE` (Billing disputes and financial reconciliation require human review and secure identity verification).
* **Historical Resolution Pattern**: Clarify difference between temporary bank pre-authorization holds and settled charges; direct to secure billing support portal.

---

### 8. `general_feedback_or_complaint`
* **Name**: General Complaints, Escalated Dissatisfaction & Agent Requests
* **Definition**: Customers expressing severe frustration, demanding a supervisor or human representative, filing complaints regarding driver/rep conduct, or making general statements without order context.
* **Representative Inquiries**:
  - *"Your customer service is the absolute worst! I need a supervisor to call me right now."*
  - *"Can someone from Amazon please DM me? I have an urgent issue."*
  - *"Who can I contact to file an official complaint against a customer rep?"*
* **Decision Boundaries**:
  - If a specific issue is named (e.g. *"I am so angry, my package is 4 days late!"*), assign the specific operational intent (`delivery_status`). Assign `general_feedback_or_complaint` only when no distinct single operational category applies or the primary intent is escalation/feedback.
* **Expected System Action**: `ESCALATE` (Human touch required for de-escalation).
* **Historical Resolution Pattern**: Empathize, apologize for the negative experience, and provide direct phone/chat callback link (`amazon.com/contact-us`).
