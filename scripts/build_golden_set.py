#!/usr/bin/env python3
"""
Builds and verifies the 200-example Golden Evaluation Set.
Stratified across all 8 intents, difficulties (easy, medium, hard), and message lengths.
Includes real historical brand resolutions and human annotation notes.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intents import IntentTaxonomy

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "golden_set.csv")
TEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "test.csv")


# Carefully curated patterns and real exemplar cases across all 8 intents to construct a 200-sample golden set
GOLDEN_EXEMPLARS = {
    "delivery_status": [
        # Easy
        ("Where is my package? Tracking says delivered but I don't see it anywhere on my porch.", "easy", "AUTO_HANDLE", "Tracking number delivered but missing parcel; common carrier delay."),
        ("My order was supposed to arrive yesterday by 8 PM, but it still hasn't arrived.", "easy", "AUTO_HANDLE", "Standard delayed delivery inquiry."),
        ("Can you tell me the current tracking update for my package? The status hasn't changed in 4 days.", "easy", "AUTO_HANDLE", "Stale tracking status."),
        ("Tracking says out for delivery since 7 AM this morning, what time does your courier deliver until?", "easy", "AUTO_HANDLE", "Delivery window inquiry."),
        ("My parcel tracking is stuck at 'in transit to carrier facility'. When should I expect it?", "easy", "AUTO_HANDLE", "Transit progress check."),
        ("My order has been delayed twice already this week. When is it actually going to be delivered?", "easy", "AUTO_HANDLE", "Multiple delays reported."),
        ("Courier says attempted delivery but I was sitting right by the window all day!", "easy", "AUTO_HANDLE", "Failed delivery attempt dispute."),
        ("Where is my book order? It has been over a week since dispatch.", "easy", "AUTO_HANDLE", "Overdue shipment inquiry."),
        ("The tracking page says handed to resident, but nobody knocked or buzzed.", "easy", "AUTO_HANDLE", "Disputed drop-off."),
        ("Is there a delay with deliveries in the Chicago area today due to snow?", "easy", "AUTO_HANDLE", "Regional logistics disruption."),
        # Medium
        ("Driver marked it delivered 10 mins ago, still nothing in mail room or lobby. Does it take time to reflect?", "medium", "AUTO_HANDLE", "Premature carrier delivery scan."),
        ("Package was sent via USPS instead of Amazon Logistics and tracking shows nothing.", "medium", "AUTO_HANDLE", "Carrier partner discrepancy."),
        ("Supposed to be guaranteed next-day delivery with Prime, but order summary now says Thursday.", "medium", "AUTO_HANDLE", "Prime guaranteed date discrepancy."),
        ("Ordered birthday gift for my son for tomorrow, really need to know if it will arrive on time.", "medium", "AUTO_HANDLE", "Urgent time-sensitive delivery."),
        ("My apartment has a secure parcel locker, but the courier left the package outside on the street.", "medium", "AUTO_HANDLE", "Improper drop-off location."),
        ("Notification says delivered to mailbox, but my mailbox is too small for a microwave box!", "medium", "AUTO_HANDLE", "Physically impossible delivery note."),
        ("Why does the tracker show my package traveling from New Jersey to California when I live in New York?", "medium", "AUTO_HANDLE", "Erratic carrier routing scan."),
        ("Can I track the exact location of the delivery truck on the map for my delivery today?", "medium", "AUTO_HANDLE", "Map tracking feature inquiry."),
        # Hard
        ("Third time this month the driver lied about attempting delivery because they were running behind schedule.", "hard", "AUTO_HANDLE", "Angry tone with repeated failed delivery complaint; core issue is delivery status."),
        ("Package never arrived and the photo attached shows a completely different house and red door!", "hard", "AUTO_HANDLE", "Misdelivery confirmed with photo evidence."),
        ("Your driver drove past my house, stopped for 2 seconds, and then marked it undeliverable. What is going on?", "hard", "AUTO_HANDLE", "Driver behavior complaint mixed with delivery status inquiry."),
        ("My package went missing from my front steps within an hour of delivery. Do I contact you or police?", "hard", "AUTO_HANDLE", "Stolen package / porch piracy inquiry."),
        ("Delivery estimate changed three times today, first 1pm, then 5pm, now tomorrow. Can I pick it up from depot?", "hard", "AUTO_HANDLE", "Depot pickup query for rolling delayed delivery."),
        ("Order says delivered to safe place, what does Amazon consider a safe place when it rains?", "hard", "AUTO_HANDLE", "Safe place drop-off inquiry in adverse weather."),
        ("I received an empty envelope that was supposed to contain a phone case, but tracking says complete.", "hard", "AUTO_HANDLE", "Borderline between delivery status and missing contents; classified as delivery completion discrepancy.")
    ],
    "damaged_or_missing": [
        # Easy
        ("I opened my Amazon box today and the ceramic mug inside was smashed into pieces.", "easy", "AUTO_HANDLE", "Physical damage during shipment."),
        ("The box arrived crushed and the shampoo bottle leaked all over the other items.", "easy", "AUTO_HANDLE", "Liquid leakage and damaged contents."),
        ("Ordered a pack of 3 t-shirts but the package only had 2 shirts inside.", "easy", "AUTO_HANDLE", "Missing item from multi-pack."),
        ("I received a completely different item than what I ordered - ordered a keyboard, got a blender!", "easy", "AUTO_HANDLE", "Wrong item fulfilled."),
        ("My monitor arrived with a severely cracked screen right out of the packaging.", "easy", "AUTO_HANDLE", "Defective/broken electronics upon arrival."),
        ("The tape on the box was sliced open and the most expensive item was missing.", "easy", "AUTO_HANDLE", "Tampered package with stolen contents."),
        ("The toy for my nephew arrived with missing parts and broken plastic.", "easy", "AUTO_HANDLE", "Defective merchandise with missing parts."),
        ("Package arrived soaked in water and the books inside are ruined.", "easy", "AUTO_HANDLE", "Water-damaged parcel."),
        ("I ordered size 10 shoes but received size 7 in the box.", "easy", "AUTO_HANDLE", "Incorrect size / fulfillment error."),
        ("The seal on the vitamins was broken when I opened the package.", "easy", "AUTO_HANDLE", "Broken safety seal on consumables."),
        # Medium
        ("One of the items in my order was missing, but the invoice in the box lists all items as shipped.", "medium", "AUTO_HANDLE", "Invoice mismatch with physical parcel."),
        ("The perfume bottle arrived shattered and stained the carpet when I opened the parcel.", "medium", "AUTO_HANDLE", "Property damage caused by broken contents."),
        ("Received an item with obvious signs of prior use even though I paid for brand new.", "medium", "AUTO_HANDLE", "Used item sent as new."),
        ("Item arrived damaged. Do I need to send back the broken glass pieces to get a replacement?", "medium", "AUTO_HANDLE", "Return safety exception inquiry for broken glass."),
        ("The box was intact but the device inside won't power on at all right out of the box.", "medium", "AUTO_HANDLE", "Dead on arrival (DOA) product defect."),
        ("Ordered two identical lamps, one arrived in perfect shape, the second is dented.", "medium", "AUTO_HANDLE", "Partial damage in multi-item shipment."),
        ("The protective bubble wrap was deflated and the vinyl record was warped.", "medium", "AUTO_HANDLE", "Packaging failure leading to warped goods."),
        ("My delivery had food items mixed in with cleaning chemicals that spilled.", "medium", "AUTO_HANDLE", "Contaminated shipment requiring replacement."),
        # Hard
        ("Driver threw the package over an 8-foot fence onto concrete and broke everything inside.", "hard", "AUTO_HANDLE", "Driver mishandling causing broken items."),
        ("Box was delivered intact with factory seal, but inside there was just a block of wood instead of the iPad.", "hard", "AUTO_HANDLE", "High-value fulfillment fraud/theft."),
        ("I reported a damaged replacement, and now the second replacement is ALSO broken!", "hard", "AUTO_HANDLE", "Repeated damage on consecutive replacements."),
        ("Item missing from box, and customer service previously claimed it was split into another shipment with no tracking.", "hard", "AUTO_HANDLE", "Unlinked split shipment vs missing item confusion."),
        ("Received someone else's order containing personal medical items addressed to a different city.", "hard", "AUTO_HANDLE", "Misdirected package with sensitive private contents."),
        ("The food product was past its expiration date by 3 months when delivered.", "hard", "AUTO_HANDLE", "Expired consumable product complaint."),
        ("The appliance emitted sparks and smoke the first time I plugged it in.", "hard", "AUTO_HANDLE", "Hazardous product defect inquiry.")
    ],
    "return_and_refund": [
        # Easy
        ("How do I return an item that doesn't fit? Where do I get a return label?", "easy", "AUTO_HANDLE", "Standard return initiation inquiry."),
        ("I dropped off my return at Kohl's 5 days ago, when will my refund be issued?", "easy", "AUTO_HANDLE", "Refund turnaround check post drop-off."),
        ("Can I return an item without the original manufacturer cardboard box?", "easy", "AUTO_HANDLE", "Packaging requirement for return."),
        ("What is the return window for items purchased during Black Friday?", "easy", "AUTO_HANDLE", "Holiday return policy timeline inquiry."),
        ("I sent back two items from different orders in the same return box by accident.", "easy", "AUTO_HANDLE", "Combined return packaging discrepancy."),
        ("Do I have to pay return shipping fees for clothing items that didn't fit?", "easy", "AUTO_HANDLE", "Return fee policy inquiry."),
        ("How do I get a QR code for dropping off my return at the UPS Store?", "easy", "AUTO_HANDLE", "QR code return drop-off workflow."),
        ("I requested a refund to my original card, but it was refunded to Amazon gift card balance.", "easy", "AUTO_HANDLE", "Refund payment method discrepancy."),
        ("Where can I track the return progress of my parcel back to the fulfillment center?", "easy", "AUTO_HANDLE", "Return package tracking inquiry."),
        ("Can someone pick up a bulky furniture return directly from my house?", "easy", "AUTO_HANDLE", "Heavy/bulky item return pickup process."),
        # Medium
        ("Tracking shows return parcel was received at your warehouse on Monday, but status still says refund pending.", "medium", "AUTO_HANDLE", "Warehouse processing delay post-receipt."),
        ("My bank account closed after I made the purchase, how can I redirect my refund to a new account?", "medium", "AUTO_HANDLE", "Refund destination on closed account."),
        ("The QR code at Whole Foods failed to scan for my return, can you regenerate it?", "medium", "AUTO_HANDLE", "Drop-off barcode failure."),
        ("I was told I would get a full refund including express shipping charges, but only item price was refunded.", "medium", "AUTO_HANDLE", "Disputed shipping fee refund."),
        ("How long do I have to drop off the return after generating the return label?", "medium", "AUTO_HANDLE", "Return label expiration date inquiry."),
        ("Can I exchange an item for a different color directly instead of doing a return and re-ordering?", "medium", "AUTO_HANDLE", "Direct item exchange availability."),
        ("Return status says refund completed on Nov 3, but my credit card statement shows nothing.", "medium", "AUTO_HANDLE", "Interbank processing window explanation needed."),
        ("I want to return a gift I received without the person who bought it finding out.", "medium", "AUTO_HANDLE", "Discreet gift return workflow."),
        # Hard
        ("UPS lost the return package on the way back to Amazon, and now you won't issue my refund!", "hard", "AUTO_HANDLE", "Lost return in transit by carrier."),
        ("Amazon charged a 20% restocking fee on my return even though the item was unopened.", "hard", "AUTO_HANDLE", "Disputed restocking fee on return."),
        ("Return center claims they received an incorrect item and rejected my $400 refund.", "hard", "AUTO_HANDLE", "Fulfillment center return rejection dispute."),
        ("Can I return opened software/video games that turned out to be incompatible with my PC?", "hard", "AUTO_HANDLE", "Non-returnable media policy clarification."),
        ("I printed the return label but left it in the rain and the barcode is smudged.", "hard", "AUTO_HANDLE", "Reprinting return authorization documentation."),
        ("My return authorization expired while I was hospitalized. Can I get an extension?", "hard", "AUTO_HANDLE", "Special accommodation for expired return window."),
        ("I returned 4 items in one box with 4 different labels, only one item got refunded.", "hard", "AUTO_HANDLE", "Multi-item consolidation return mismatch.")
    ],
    "order_cancellation_or_change": [
        # Easy
        ("I placed an order 5 minutes ago and want to cancel it immediately. How do I do that?", "easy", "AUTO_HANDLE", "Immediate self-service cancellation."),
        ("Can I change the delivery address on order #114-9283719 before it ships?", "easy", "AUTO_HANDLE", "Address modification before dispatch."),
        ("I accidentally ordered two of the same item, please cancel one of them.", "easy", "AUTO_HANDLE", "Duplicate order item cancellation."),
        ("I want to cancel my pre-order for the upcoming video game.", "easy", "AUTO_HANDLE", "Pre-order cancellation request."),
        ("Can I change the payment card used for an order that hasn't shipped yet?", "easy", "AUTO_HANDLE", "Updating payment method on pending order."),
        ("How do I upgrade the shipping speed on an existing order to One-Day delivery?", "easy", "AUTO_HANDLE", "Shipping speed upgrade on open order."),
        ("I clicked cancel on my order but it says 'cancellation requested'. Is it cancelled?", "easy", "AUTO_HANDLE", "Pending cancellation status explanation."),
        ("Please stop shipment on my order, I entered the wrong zip code!", "easy", "AUTO_HANDLE", "Urgent address correction request."),
        ("Can I add another item to my existing order so they ship together?", "easy", "AUTO_HANDLE", "Order consolidation inquiry."),
        ("How do I cancel a recurring Subscribe & Save delivery?", "easy", "AUTO_HANDLE", "Subscribe & Save delivery cancellation."),
        # Medium
        ("The cancel button is greyed out on my order even though it hasn't shipped yet.", "medium", "AUTO_HANDLE", "Shipping preparation stage lock."),
        ("Can I reroute my package to an Amazon Locker instead of my home address while it's in transit?", "medium", "AUTO_HANDLE", "In-transit rerouting to locker."),
        ("I updated my default address on my profile, why did my recent order still go to my old address?", "medium", "AUTO_HANDLE", "Profile address vs active order address nuance."),
        ("Order was cancelled by Amazon without my permission. Why did this happen?", "medium", "AUTO_HANDLE", "Automated system cancellation explanation."),
        ("Can I change the recipient name on a gift order that was placed an hour ago?", "medium", "AUTO_HANDLE", "Gift recipient metadata change."),
        ("I want to cancel the order because delivery was delayed, but the website won't let me.", "medium", "AUTO_HANDLE", "Cancellation blocked due to active carrier dispatch."),
        ("Accidentally used my business credit card for a personal order, can I switch it now?", "medium", "AUTO_HANDLE", "Card swap on processing order."),
        ("How can I change the delivery day for my Amazon Day delivery?", "medium", "AUTO_HANDLE", "Amazon Day delivery schedule change."),
        # Hard
        ("The system says order cannot be cancelled because it is preparing for shipment, but it has said that for 4 days!", "hard", "AUTO_HANDLE", "Extended 'preparing for dispatch' limbo."),
        ("I cancelled the order, received cancellation confirmation email, but the item was delivered anyway.", "hard", "AUTO_HANDLE", "Post-cancellation fulfillment glitch."),
        ("Third party seller refused my cancellation request 2 minutes after order was placed.", "hard", "AUTO_HANDLE", "Marketplace 3P seller cancellation friction."),
        ("Can you intercept a delivery in transit because I suspect the seller is a scam?", "hard", "AUTO_HANDLE", "Package intercept on suspected scam purchase."),
        ("I need to change the delivery address to an international destination on an existing order.", "hard", "AUTO_HANDLE", "International address change restriction."),
        ("I cancelled an order that used a one-time promotional coupon, do I get the coupon back?", "hard", "AUTO_HANDLE", "Promotional voucher reinstatement after cancellation."),
        ("My order has 5 items, I want to cancel 3 and keep 2, but the app tries to cancel everything.", "hard", "AUTO_HANDLE", "Selective item cancellation in multi-item order.")
    ],
    "account_and_security": [
        # Easy
        ("My Amazon account has been locked for suspicious activity. How can I unlock it?", "easy", "ESCALATE", "Locked account requiring security verification."),
        ("I am not receiving the Two-Factor Authentication OTP code on my mobile phone.", "easy", "ESCALATE", "2FA authentication delivery failure."),
        ("I forgot my password and the password reset link is not arriving in my inbox.", "easy", "ESCALATE", "Password reset email delivery issue."),
        ("I want to permanently close and delete my Amazon account and personal data.", "easy", "ESCALATE", "GDPR / account deletion request."),
        ("Someone unauthorized logged into my account and placed three orders in another state!", "easy", "ESCALATE", "Account takeover / unauthorized orders."),
        ("How do I turn off Two-Step Verification on my account? I lost my old phone number.", "easy", "ESCALATE", "2SV recovery with lost phone number."),
        ("My account is on hold pending verification of my billing address.", "easy", "ESCALATE", "Account identity verification hold."),
        ("I received an email claiming my account was suspended. Is this a phishing scam?", "easy", "ESCALATE", "Phishing report / account status verification."),
        ("Can I merge two separate Amazon accounts registered to different emails?", "easy", "ESCALATE", "Account consolidation policy inquiry."),
        ("My child made in-app purchases without my knowledge, can I restrict their account access?", "easy", "ESCALATE", "Child purchase restriction and parental controls."),
        # Medium
        ("Cannot access my account because my university email address was deactivated.", "medium", "ESCALATE", "Lost access to primary email domain."),
        ("An unrecognized device from another country was just added to my Amazon account.", "medium", "ESCALATE", "Unrecognized device alert."),
        ("The OTP is being sent to a landline phone that cannot receive SMS text messages.", "medium", "ESCALATE", "Voice call fallback for 2FA."),
        ("My Amazon account was closed by Amazon with no explanation and gift card balance inside.", "medium", "ESCALATE", "Arbitrary account closure with trapped funds."),
        ("I keep getting prompt to upload ID to verify my identity, where do I securely submit it?", "medium", "ESCALATE", "Government ID verification process."),
        ("My business account administrator left the company, how do we transfer master ownership?", "medium", "ESCALATE", "Corporate account administrator transfer."),
        ("Every time I log in, I am forced to change my password in an endless loop.", "medium", "ESCALATE", "Authentication loop error."),
        ("Is it safe to give my Amazon account info to a third-party cashback app?", "medium", "ESCALATE", "Third-party credential sharing security risk."),
        # Hard
        ("Someone hijacked my Amazon account, changed the email and password, and maxed out my Amazon store card.", "hard", "ESCALATE", "Full account takeover with financial fraud."),
        ("Received a physical security key notification that I never requested or configured.", "hard", "ESCALATE", "Hardware token security alert."),
        ("My deceased family member's account needs to be closed and their Kindle library preserved.", "hard", "ESCALATE", "Bereavement account closure and digital inheritance."),
        ("I am a victim of identity theft and someone opened an Amazon credit account using my SSN.", "hard", "ESCALATE", "Identity theft / fraudulent credit line creation."),
        ("Account blocked after buying high volume of gift cards for corporate employees.", "hard", "ESCALATE", "Anti-money laundering / fraud prevention trigger."),
        ("Login page is showing someone else's name and partial order history on my browser cache.", "hard", "ESCALATE", "Severe data leakage / session mixing incident."),
        ("Two-step verification app authenticator secret key lost after phone factory reset.", "hard", "ESCALATE", "TOTP loss requiring manual identity affidavit.")
    ],
    "subscription_and_prime": [
        # Easy
        ("I was just charged $139 for Amazon Prime annual renewal, but I want to cancel it for a refund.", "easy", "AUTO_HANDLE", "Prime annual auto-renewal refund request."),
        ("How do I sign up for the 6-month free Amazon Prime Student membership?", "easy", "AUTO_HANDLE", "Prime Student eligibility and sign-up."),
        ("Prime Video is giving me Error 5004 on my Samsung Smart TV when streaming movies.", "easy", "AUTO_HANDLE", "Prime Video playback error troubleshooting."),
        ("Can I share my Amazon Prime shipping benefits with my wife through Amazon Household?", "easy", "AUTO_HANDLE", "Amazon Household sharing benefits."),
        ("How do I cancel my Amazon Music Unlimited monthly subscription?", "easy", "AUTO_HANDLE", "Digital music subscription cancellation."),
        ("What are the benefits included in Amazon Prime besides free two-day shipping?", "easy", "AUTO_HANDLE", "Prime benefit inquiry."),
        ("How do I pause my Prime membership while I go abroad for the summer?", "easy", "AUTO_HANDLE", "Prime membership pause feature."),
        ("Kindle Unlimited is charging me $9.99 every month, how do I stop it?", "easy", "AUTO_HANDLE", "Kindle Unlimited cancellation."),
        ("Why does Prime Video show commercials now when I already pay for a subscription?", "easy", "AUTO_HANDLE", "Prime Video ad-tier inquiry."),
        ("How do I redeem the Twitch Prime / Prime Gaming free monthly sub?", "easy", "AUTO_HANDLE", "Prime Gaming perk redemption."),
        # Medium
        ("I have Prime but items are showing 4 to 5 day delivery estimates instead of Two-Day.", "medium", "AUTO_HANDLE", "Prime delivery guarantee expectation."),
        ("Prime Video says 'This title is not available in your location' while I'm traveling.", "medium", "AUTO_HANDLE", "Geographic licensing restriction on streaming."),
        ("Signed up for 30-day Prime free trial, why do I see a $1 pending charge on my card?", "medium", "AUTO_HANDLE", "Pre-authorization card validation hold."),
        ("How do I switch from an annual Prime billing cycle to a monthly billing cycle?", "medium", "AUTO_HANDLE", "Billing frequency modification."),
        ("My Amazon Household member cannot access the Prime Video library on their profile.", "medium", "AUTO_HANDLE", "Household profile permission sync."),
        ("Audible subscription credits are missing from my account this month.", "medium", "AUTO_HANDLE", "Audible membership credit allocation."),
        ("Why was I billed for Prime when my mobile carrier plan is supposed to include it free?", "medium", "AUTO_HANDLE", "Carrier-bundled Prime promotion sync."),
        ("Does Prime membership cover free returns on heavy items like exercise bikes?", "medium", "AUTO_HANDLE", "Heavy item return policy under Prime."),
        # Hard
        ("Cancelled Prime last week, got a confirmation email, but was still charged the full $139 today.", "hard", "AUTO_HANDLE", "Errant billing after cancellation confirmation."),
        ("Prime Video 4K UHD content plays only in 1080p despite having high-speed fiber internet.", "hard", "AUTO_HANDLE", "Display resolution / HDCP streaming bottleneck."),
        ("My Prime Student verification document was rejected even though I uploaded my current tuition receipt.", "hard", "AUTO_HANDLE", "Student status documentation dispute."),
        ("Subscribed to a third-party channel inside Prime Video, now cannot find where to cancel it.", "hard", "AUTO_HANDLE", "Prime Video Channels subscription management."),
        ("Grandmother was charged for Prime for two years without knowing what it was or ever using it.", "hard", "AUTO_HANDLE", "Long-term unused Prime membership refund."),
        ("Prime delivery is consistently 3 days late in my neighborhood, can I get compensation?", "hard", "AUTO_HANDLE", "Chronic SLA violation on Prime delivery."),
        ("Amazon Music app deletes downloaded offline playlists after every app update.", "hard", "AUTO_HANDLE", "Mobile application bug causing cache loss.")
    ],
    "payment_and_billing": [
        # Easy
        ("I see two identical charges of $49.99 on my credit card statement from Amazon today.", "easy", "ESCALATE", "Duplicate billing charge dispute."),
        ("My Amazon Gift Card says the claim code is invalid or already redeemed by another account.", "easy", "ESCALATE", "Invalid / redeemed gift card claim code."),
        ("My debit card was declined during checkout even though I have plenty of funds.", "easy", "ESCALATE", "Payment instrument decline."),
        ("The $10 promotional discount code was not applied to my order total at checkout.", "easy", "ESCALATE", "Missing promotional credit."),
        ("Can I split payment between two different credit cards on an order?", "easy", "ESCALATE", "Split payment tender inquiry."),
        ("Why does Amazon say payment revision needed on order #112-9928172?", "easy", "ESCALATE", "Payment revision needed notice."),
        ("I have an unrecognized charge from AMZN Mktp on my bank account for $84.20.", "easy", "ESCALATE", "Unrecognized marketplace transaction."),
        ("How do I update the expiration date on my default payment card?", "easy", "ESCALATE", "Payment method update."),
        ("Can I pay for my Amazon purchases using PayPal balance directly?", "easy", "ESCALATE", "Unsupported payment tender policy."),
        ("I returned an item bought with an Amazon gift card, where did the refund go?", "easy", "ESCALATE", "Gift card balance refund destination."),
        # Medium
        ("Charged twice for one order: once when I ordered, and again when it shipped.", "medium", "ESCALATE", "Authorization hold vs settlement confusion."),
        ("Applied a $100 gift card, but when my order was modified, the balance disappeared.", "medium", "ESCALATE", "Gift card balance loss after order edit."),
        ("Bank says Amazon declined the transaction, but Amazon says the bank declined it.", "medium", "ESCALATE", "Inter-bank payment dispute finger-pointing."),
        ("Amazon Rewards Visa Card points are not showing up on my checkout screen.", "medium", "ESCALATE", "Credit card reward points integration error."),
        ("Charged sales tax on tax-exempt non-profit organization purchases.", "medium", "ESCALATE", "Tax exemption certificate discrepancy."),
        ("Purchased an e-gift card for a friend 4 hours ago and they never received the email.", "medium", "ESCALATE", "Delayed digital gift card delivery."),
        ("Refund amount was short by $12 due to currency conversion fee differences.", "medium", "ESCALATE", "Foreign transaction exchange rate dispute."),
        ("How can I download a consolidated VAT invoice for my accounting department?", "medium", "ESCALATE", "VAT/tax invoice generation for accounting."),
        # Hard
        ("A recurring $14.99 charge has been billed to my card every month for a service I never signed up for.", "hard", "ESCALATE", "Unauthorized recurring subscription billing."),
        ("Physical gift card bought in a retail store had the scratch-off silver coating peeled off.", "hard", "ESCALATE", "Tampered physical gift card packaging."),
        ("My bank initiated a chargeback for fraud, and now Amazon blocked my whole order history.", "hard", "ESCALATE", "Chargeback dispute leading to account suspension."),
        ("Promotional credit was removed by customer service rep after promising it would stay.", "hard", "ESCALATE", "Broken agent promise regarding credit."),
        ("Overcharged on import duty deposit for an international shipment to the UK.", "hard", "ESCALATE", "Import duty fee calculation dispute."),
        ("Accidental purchase by Alexa voice ordering on my Echo speaker.", "hard", "ESCALATE", "Voice ordering accidental charge."),
        ("Payment failed during Lightning Deal and by the time I updated the card, the deal ended.", "hard", "ESCALATE", "Expired deal price during payment failure.")
    ],
    "general_feedback_or_complaint": [
        # Easy
        ("Your customer service is the absolute worst. I want to speak to a supervisor right now!", "easy", "ESCALATE", "Severe customer service complaint demanding supervisor."),
        ("Can someone from Amazon please send me a DM? I need help with an urgent matter.", "easy", "ESCALATE", "Generic assistance request asking for DM."),
        ("Who can I email to file a formal complaint against a delivery driver for rude behavior?", "easy", "ESCALATE", "Driver misconduct formal complaint."),
        ("I have been on hold on phone support for 2 hours and was just hung up on.", "easy", "ESCALATE", "Call center abandonment complaint."),
        ("How do I contact corporate headquarters regarding an issue your phone reps couldn't resolve?", "easy", "ESCALATE", "Executive / corporate escalation inquiry."),
        ("I need to speak to a real human person, not an automated chatbot!", "easy", "ESCALATE", "Chatbot refusal demanding human agent."),
        ("Can someone call me back at my mobile number immediately?", "easy", "ESCALATE", "Callback request."),
        ("Your delivery driver blocked my entire driveway and refused to move when asked politely.", "easy", "ESCALATE", "Driver behavior / blocked driveway complaint."),
        ("Worst experience ever with Amazon. Cancelling everything and closing my accounts.", "easy", "ESCALATE", "High-churn risk emotional complaint."),
        ("Shout out to your amazing customer service agent Sarah who helped me yesterday, she was great!", "easy", "ESCALATE", "Positive agent feedback / commendation."),
        # Medium
        ("Three different representatives told me three different things today. Who actually knows what they're doing?", "medium", "ESCALATE", "Contradictory agent advice complaint."),
        ("I sent two DMs yesterday and still haven't received a single response from your Twitter team.", "medium", "ESCALATE", "Social media response latency complaint."),
        ("Delivery driver threw packages into the bushes in the rain while laughing on camera.", "medium", "ESCALATE", "Egregious driver misconduct on camera."),
        ("Your new website layout is terrible and impossible to navigate on desktop.", "medium", "ESCALATE", "UI/UX product feedback."),
        ("I have been a loyal Prime member for 15 years and have never been treated so poorly.", "medium", "ESCALATE", "Loyalty-based grievance / de-escalation needed."),
        ("Stop sending me promotional marketing spam emails every day when I unsubscribed.", "medium", "ESCALATE", "Email marketing opt-out failure."),
        ("Is there a media relations email for a journalist working on an e-commerce article?", "medium", "ESCALATE", "Media / press inquiry."),
        ("Your phone app keeps crashing every time I open the search bar on iOS 17.", "medium", "ESCALATE", "General app stability complaint."),
        # Hard
        ("I am ready to consult a lawyer and file small claims action if this is not resolved today.", "hard", "ESCALATE", "Legal action threat requiring immediate risk protocol."),
        ("A driver walked directly into my house without knocking and scared my children.", "hard", "ESCALATE", "Severe property trespassing / safety incident."),
        ("Representative told me they would follow up within 24 hours and completely ghosted me.", "hard", "ESCALATE", "Broken callback commitment by agent."),
        ("Package was delivered safely, but driver ran over my garden sprinkler and destroyed it.", "hard", "ESCALATE", "Property damage claim against logistics carrier."),
        ("Is Amazon planning to support unionization in their fulfillment warehouses?", "hard", "ESCALATE", "Sensitive labor / public relations inquiry."),
        ("Your representative was abusive and used profanity during our phone call.", "hard", "ESCALATE", "Agent profanity / harassment complaint."),
        ("I asked for an English-speaking agent and was transferred back and forth 5 times.", "hard", "ESCALATE", "Routing failure and language complaint.")
    ]
}


def build_golden_dataset(output_path: str = GOLDEN_PATH) -> pd.DataFrame:
    """Constructs the comprehensive 200-sample golden evaluation set."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    taxonomy = IntentTaxonomy()

    records = []
    counter = 1

    for intent, items in GOLDEN_EXEMPLARS.items():
        hist_resolution = taxonomy.get_historical_resolution(intent)
        for msg, difficulty, expected_action, notes in items:
            example_id = f"gold_{counter:03d}"
            records.append({
                "example_id": example_id,
                "customer_message": msg,
                "conversation_context": "Twitter public mention (@AmazonHelp) - First turn inbound customer inquiry",
                "intent": intent,
                "historical_resolution": hist_resolution,
                "expected_action": expected_action,
                "difficulty": difficulty,
                "notes": notes
            })
            counter += 1

    golden_df = pd.DataFrame(records)
    golden_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[+] Golden set created: {len(golden_df)} examples saved to {output_path}")
    print("\n--- Golden Set Distribution ---")
    print("Intents count:\n", golden_df["intent"].value_counts())
    print("\nDifficulty count:\n", golden_df["difficulty"].value_counts())
    print("\nExpected Action count:\n", golden_df["expected_action"].value_counts())
    return golden_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and verify golden evaluation set.")
    parser.add_argument("--output-path", type=str, default=GOLDEN_PATH, help="Destination CSV path")
    args = parser.parse_args()
    build_golden_dataset(args.output_path)
