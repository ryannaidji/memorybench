"""Generate the synthetic golden set (deterministic, seed=7).

Each customer has 3 support conversations. Every conversation mixes 1-3 durable facts
(the labels) with small talk and transient details that must NOT be extracted.
Outputs:
  evals/golden.jsonl   one conversation per line with its gold facts
  evals/queries.jsonl  retrieval queries per customer with the gold fact they should hit
"""
import json
import random
from pathlib import Path

R = random.Random(7)
HERE = Path(__file__).parent

NAMES = ["Maria", "James", "Aisha", "Luca", "Chen", "Sofia", "Omar", "Elena", "David", "Priya",
         "Tom", "Yuki", "Carlos", "Nina", "Ahmed", "Grace", "Ivan", "Lea", "Marcus", "Fatima"]

# (customer phrasing, gold value, retrieval query)
CONTACT = [
    ("Please email me instead of calling, I can't take calls at work.", "Prefers email over phone calls",
     "How should we contact this customer?"),
    ("Text me if anything changes, I never check my email.", "Prefers SMS; rarely checks email",
     "What channel does the customer prefer?"),
    ("Only call me after 6pm Eastern, I'm in meetings all day.", "Wants calls only after 6pm Eastern",
     "When can we call the customer?"),
    ("WhatsApp is the best way to reach me.", "Prefers to be contacted on WhatsApp",
     "How should we contact this customer?"),
]
PLAN = [
    ("We're on the Pro plan with 25 seats.", "Has the Pro plan with 25 seats", "What plan is the customer on?"),
    ("I just upgraded to Enterprise last week.", "Upgraded to the Enterprise plan", "What plan is the customer on?"),
    ("I'm still on the free tier for now.", "Uses the free tier", "Which subscription does the customer have?"),
    ("We pay annually for the Business plan.", "Has the Business plan, billed annually",
     "Which subscription does the customer have?"),
]
ISSUE = [
    ("Since the last update, the mobile app crashes every time I open the reports tab.",
     "Mobile app crashes when opening the reports tab", "What problems has the customer reported?"),
    ("I was charged twice for October.", "Was charged twice for October", "Any billing problems?"),
    ("Our SSO login with Okta keeps failing for new users.", "Okta SSO login fails for new users",
     "What problems has the customer reported?"),
    ("The CSV export cuts off after 10,000 rows.", "CSV export truncates after 10,000 rows",
     "Any data export issues?"),
    ("Webhooks to our server have been timing out since Monday.", "Webhooks time out since Monday",
     "What problems has the customer reported?"),
]
PREF = [
    ("Could you send invoices as PDF? Our accounting needs that.", "Wants invoices as PDF", "How does the customer want invoices?"),
    ("I'd rather be helped in Spanish if possible.", "Prefers support in Spanish", "What language does the customer prefer?"),
    ("Please keep replies short, bullet points are best for me.", "Prefers short bullet-point replies",
     "How does the customer like answers formatted?"),
    ("I prefer the dark theme, the light one hurts my eyes.", "Prefers the dark theme", "Any UI preferences?"),
]
PROFILE = [
    ("I'm the IT admin for a 40-person dental clinic.", "IT admin at a 40-person dental clinic", "Who is this customer?"),
    ("We're a small e-commerce shop based in Miami.", "Runs a small e-commerce shop in Miami", "What business is the customer in?"),
    ("I run the data team at a logistics company.", "Leads the data team at a logistics company", "Who is this customer?"),
    ("Our nonprofit has about 12 volunteers using the app.", "Nonprofit with about 12 volunteer users",
     "What business is the customer in?"),
]
POOLS = {"contact_preference": CONTACT, "plan": PLAN, "issue": ISSUE, "preference": PREF, "profile": PROFILE}

NOISE = [  # transient details: should NOT become memory
    "By the way, the weather here is terrible today.",
    "Sorry, my coffee just spilled, give me a second.",
    "Thanks for picking up so fast!",
    "I'm typing from my phone so excuse typos.",
    "Ha, that's funny. Anyway.",
]
AGENT_OPEN = ["Hi {n}, thanks for contacting support. How can I help?", "Hello {n}! What can I do for you today?"]
AGENT_ACK = ["Got it, I've noted that.", "Thanks, understood.", "Okay, I'll make sure the team knows.", "Noted."]
AGENT_CLOSE = ["Anything else I can help with?", "Is there anything else today?"]


def conversation(name, facts):
    turns = [{"role": "agent", "text": R.choice(AGENT_OPEN).format(n=name)}]
    for i, (phrase, _, _) in enumerate(facts):
        if R.random() < 0.4:
            turns.append({"role": "customer", "text": R.choice(NOISE)})
            turns.append({"role": "agent", "text": "No problem!"})
        turns.append({"role": "customer", "text": phrase})
        turns.append({"role": "agent", "text": R.choice(AGENT_ACK)})
    turns.append({"role": "agent", "text": R.choice(AGENT_CLOSE)})
    turns.append({"role": "customer", "text": R.choice(["No, that's all. Thanks!", "That's it, bye.", "All good, thank you."])})
    return turns


def main():
    golden, queries = [], []
    for c, name in enumerate(NAMES):
        cid = f"cust_{c:02d}"
        cats = list(POOLS)
        R.shuffle(cats)
        chosen = {cat: R.choice(POOLS[cat]) for cat in cats}  # one fact per category per customer
        # split the 5 facts over 3 conversations: 2 / 2 / 1
        groups = [cats[0:2], cats[2:4], cats[4:5]]
        for k, group in enumerate(groups):
            facts = [chosen[cat] for cat in group]
            golden.append({"conv_id": f"{cid}_c{k}", "customer_id": cid, "turns": conversation(name, facts),
                           "facts": [{"category": cat, "value": chosen[cat][1]} for cat in group]})
        for cat in cats:
            queries.append({"customer_id": cid, "query": chosen[cat][2], "category": cat, "expected": chosen[cat][1]})
    (HERE / "golden.jsonl").write_text("\n".join(json.dumps(g) for g in golden) + "\n", encoding="utf-8")
    (HERE / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    print(f"{len(golden)} conversations, {sum(len(g['facts']) for g in golden)} gold facts, {len(queries)} queries")


if __name__ == "__main__":
    main()
