"""Generate a fine-tuning set that does not overlap the eval golden set.

- Different seed (11 vs 7) and different customers.
- The LAST phrasing template of every category is held out: it never appears in training,
  only in the eval golden set, so the eval also measures generalisation to unseen phrasings.

    python -m evals.gen_train --customers 150 --out finetune/train.jsonl
Each line: {"messages": [system, user, assistant]} in the exact format used at inference.
"""
import argparse
import json
import random
from pathlib import Path

from evals import gen_golden as g
from memorybench.extract import PROMPT_V2, render
from memorybench.schemas import Conversation

HELD_OUT = {cat: pool[-1][1] for cat, pool in g.POOLS.items()}


def build(n_customers: int, seed: int = 11):
    g.R = random.Random(seed)
    rows = []
    pools = {cat: pool[:-1] for cat, pool in g.POOLS.items()}  # drop held-out template
    for c in range(n_customers):
        name = g.R.choice(g.NAMES)
        cats = list(pools)
        g.R.shuffle(cats)
        k = g.R.choice([1, 2, 2, 3])
        chosen = cats[:k]
        facts = [g.R.choice(pools[cat]) for cat in chosen]
        if g.R.random() < 0.1:  # some conversations with nothing durable to remember
            chosen, facts = [], []
        turns = g.conversation(name, facts) if facts else [
            {"role": "agent", "text": "Hi, how can I help?"},
            {"role": "customer", "text": g.R.choice(g.NOISE)},
            {"role": "agent", "text": "No problem!"},
            {"role": "customer", "text": "Actually never mind, all good. Bye!"}]
        conv = Conversation(customer_id=f"train_{c:03d}", turns=turns)
        target = {"facts": [{"category": cat, "value": f[1]} for cat, f in zip(chosen, facts)]}
        rows.append({"messages": [
            {"role": "system", "content": PROMPT_V2},
            {"role": "user", "content": f"Conversation:\n{render(conv)}\n\nReturn the facts as JSON."},
            {"role": "assistant", "content": json.dumps(target)}]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="finetune/train.jsonl")
    a = ap.parse_args()
    rows = build(a.customers, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} training conversations -> {a.out}; held-out templates: {list(HELD_OUT.values())}")


if __name__ == "__main__":
    main()
