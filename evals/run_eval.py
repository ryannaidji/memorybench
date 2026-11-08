"""Evaluate extraction (precision / recall / F1) and retrieval (recall@k, MRR) on the golden set.

    python -m evals.run_eval --model qwen2.5:3b [--limit 30] [--out evals/results/qwen2.5-3b.json]
    python -m evals.run_eval --gate evals/thresholds.json   # exit 1 if below thresholds (CI)

Matching: an extracted fact matches a gold fact when the categories are equal and the
values are semantically equivalent (cosine similarity of embeddings >= SIM_THRESHOLD).
Each gold fact can be matched at most once (greedy by similarity).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from memorybench import llm
from memorybench.extract import extract
from memorybench.schemas import Conversation, Fact
from memorybench.store import FactStore

HERE = Path(__file__).parent
SIM_THRESHOLD = 0.80


def load(name):
    return [json.loads(l) for l in (HERE / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def match(pred: list[Fact], gold: list[dict]) -> int:
    if not pred or not gold:
        return 0
    pv = llm.embed([p.value for p in pred])
    gv = llm.embed([g["value"] for g in gold])
    sims = pv @ gv.T
    pairs = sorted(((sims[i, j], i, j) for i in range(len(pred)) for j in range(len(gold))
                    if pred[i].category.value == gold[j]["category"] and sims[i, j] >= SIM_THRESHOLD), reverse=True)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j)
    return len(used_g)


def evaluate(model: str, limit: int | None, prompt: str = "v2"):
    golden = load("golden.jsonl")
    if limit:
        golden = golden[:limit]
    tp = n_pred = n_gold = 0
    latencies, parse_errors = [], 0
    store = FactStore(":memory:")
    per_conv = []
    for g in golden:
        conv = Conversation(customer_id=g["customer_id"], turns=g["turns"])
        pred, info = extract(conv, model=model, prompt=prompt)
        latencies.append(info["latency_ms"])
        parse_errors += bool(info["parse_error"])
        hit = match(pred, g["facts"])
        tp += hit; n_pred += len(pred); n_gold += len(g["facts"])
        store.add(g["customer_id"], pred)
        per_conv.append({"conv_id": g["conv_id"], "gold": len(g["facts"]), "pred": len(pred), "tp": hit,
                         "pred_values": [f"{p.category.value}: {p.value}" for p in pred]})
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Retrieval over what the extractor actually stored (end-to-end)
    customers = {g["customer_id"] for g in golden}
    queries = [q for q in load("queries.jsonl") if q["customer_id"] in customers]
    hits5, rr = 0, []
    for q in queries:
        res = store.recall(q["customer_id"], q["query"], k=5)
        rank = None
        if res:
            ev = llm.embed([q["expected"]])[0]
            rv = llm.embed([h.fact.value for h in res])
            for r, s in enumerate(rv @ ev, start=1):
                if s >= SIM_THRESHOLD:
                    rank = r
                    break
        hits5 += rank is not None
        rr.append(1 / rank if rank else 0.0)
    lat = sorted(latencies)
    return {
        "model": model, "prompt": prompt, "conversations": len(golden), "gold_facts": n_gold, "predicted_facts": n_pred,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "retrieval_recall_at_5": round(hits5 / len(queries), 3) if queries else 0.0,
        "retrieval_mrr": round(statistics.mean(rr), 3) if rr else 0.0,
        "latency_ms_p50": lat[len(lat) // 2], "latency_ms_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        "parse_errors": parse_errors, "per_conversation": per_conv,
        "date": time.strftime("%Y-%m-%d"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=llm.settings()["model"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--out")
    ap.add_argument("--gate", help="thresholds JSON; exit 1 if any metric is below")
    a = ap.parse_args()
    res = evaluate(a.model, a.limit, a.prompt)
    summary = {k: v for k, v in res.items() if k != "per_conversation"}
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if a.gate:
        th = json.loads(Path(a.gate).read_text())
        failed = {k: (res[k], v) for k, v in th.items() if res[k] < v}
        if failed:
            print("EVAL GATE FAILED:", failed)
            sys.exit(1)
        print("EVAL GATE PASSED")


if __name__ == "__main__":
    main()
