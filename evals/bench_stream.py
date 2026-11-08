"""Streaming benchmarks (real Redpanda + PostgreSQL + Ollama). Needs `docker compose up -d redpanda postgres`.

    python -m evals.bench_stream scale --consumers 1 2 3 --n 30
    python -m evals.bench_stream idempotency --n 20
    python -m evals.bench_stream failure --n 30 --kill-after 8

Each consumer is a separate OS process in one consumer group. Results + per-consumer JSON logs go to
evals/results/stream/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from memorybench.pgstore import PostgresFactStore
from memorybench.stream import BOOTSTRAP, DLQ_TOPIC, TOPIC, producer

OUT = Path(__file__).parent / "results" / "stream"
MODEL = os.getenv("LLM_MODEL", "memorybench-qwen1.5b-ft")


def warmup():
    from memorybench.extract import extract
    from memorybench.schemas import Conversation
    os.environ["LLM_MODEL"] = MODEL
    extract(Conversation(customer_id="w", turns=[{"role": "customer", "text": "Please email me."}]))


def reset(store, partitions=3):
    producer.ensure_topics(partitions=partitions, recreate=True)
    store.reset()


def start_consumers(k, group, logdir: Path):
    procs = []
    env = {**os.environ, "STORE": "postgres", "LLM_MODEL": MODEL, "PYTHONUNBUFFERED": "1"}
    for i in range(k):
        f = open(logdir / f"consumer-{i}.log", "w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-m", "memorybench.stream.consumer", "--id", f"c{i}", "--group", group],
                             stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((p, f))
    # wait until the group has assigned all partitions
    deadline = time.time() + 60
    while time.time() < deadline:
        parts = set()
        for i in range(k):
            for line in (logdir / f"consumer-{i}.log").read_text(encoding="utf-8").splitlines():
                if '"assigned"' in line:
                    parts |= set(json.loads(line)["partitions"])
        if len(parts) == 3 and all('"assigned"' in (logdir / f"consumer-{i}.log").read_text(encoding="utf-8") for i in range(k)):
            break
        time.sleep(1)
    time.sleep(3)
    return procs


def stop_consumers(procs):
    for p, f in procs:
        if p.poll() is None:
            p.terminate()
    for p, f in procs:
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
        f.close()


def log_counts(logdir: Path) -> dict:
    c = {"processed": 0, "duplicate_skipped": 0, "sent_to_dlq": 0}
    for lf in logdir.glob("consumer-*.log"):
        for line in lf.read_text(encoding="utf-8").splitlines():
            try:
                msg = json.loads(line).get("msg")
            except json.JSONDecodeError:
                continue
            if msg in c:
                c[msg] += 1
    return c


def wait_for(cond, timeout=1800, every=1.0, on_tick=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if on_tick:
            on_tick()
        if cond():
            return True
        time.sleep(every)
    return False


def metrics(store, n_produced) -> dict:
    rows = store.con.execute("""SELECT produced_at, processed_at, consumer_id, extract_ms, kafka_partition
                                FROM processed_events WHERE produced_at IS NOT NULL""").fetchall()
    lat = np.array([(r[1] - r[0]).total_seconds() for r in rows])
    span = (max(r[1] for r in rows) - min(r[0] for r in rows)).total_seconds()
    per_consumer, per_partition = {}, {}
    for r in rows:
        per_consumer[r[2]] = per_consumer.get(r[2], 0) + 1
        per_partition[r[4]] = per_partition.get(r[4], 0) + 1
    return {"produced": n_produced, "processed_distinct": len(rows), "facts": store.count_facts(),
            "wall_s": round(span, 1), "throughput_conv_per_min": round(len(rows) / span * 60, 2),
            "e2e_latency_p50_s": round(float(np.percentile(lat, 50)), 2), "e2e_latency_p95_s": round(float(np.percentile(lat, 95)), 2),
            "extract_ms_p50": int(np.percentile([r[3] for r in rows], 50)),
            "per_consumer": dict(sorted(per_consumer.items())), "per_partition": dict(sorted(per_partition.items()))}


def run_scale(store, k, n, tag):
    logdir = OUT / f"{tag}_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    reset(store)
    procs = start_consumers(k, f"bench-{tag}-{int(time.time())}", logdir)
    try:
        producer.produce(producer.golden_events(tag, n))
        ok = wait_for(lambda: store.count_events() >= n)
    finally:
        stop_consumers(procs)
    return {"consumers": k, "partitions": 3, "completed": ok, **metrics(store, n), "log_counts": log_counts(logdir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scale", "idempotency", "failure"])
    ap.add_argument("--consumers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--kill-after", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    store = PostgresFactStore()
    warmup()
    meta = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL, "bootstrap": BOOTSTRAP,
            "topic": TOPIC, "ollama_num_parallel": os.getenv("OLLAMA_NUM_PARALLEL", "unset (Ollama default)"),
            "machine": "single laptop, CPU only (Intel, 8 cores), Docker Desktop: Redpanda --smp 1 --memory 512M, pgvector/pg16"}

    if a.mode == "scale":
        runs = []
        for k in a.consumers:
            r = run_scale(store, k, a.n, f"scale_k{k}")
            print(json.dumps(r))
            runs.append(r)
        result = {**meta, "n_conversations": a.n, "runs": runs}

    elif a.mode == "idempotency":
        logdir = OUT / "idempotency_logs"
        logdir.mkdir(parents=True, exist_ok=True)
        reset(store)
        procs = start_consumers(1, f"bench-idem-{int(time.time())}", logdir)
        try:
            evs = producer.golden_events("idem", a.n)
            producer.produce(evs)
            wait_for(lambda: store.count_events() >= a.n)
            first = {"events": store.count_events(), "facts": store.count_facts()}
            # 1) replay the SAME events (same event_id): ledger short-circuits, no LLM call
            producer.produce(evs)
            wait_for(lambda: log_counts(logdir)["duplicate_skipped"] >= a.n, timeout=300)
            replay_same = {"events": store.count_events(), "facts": store.count_facts(), **log_counts(logdir)}
            # 2) replay the same conversations with NEW event ids: re-extracted, UNIQUE constraint absorbs the facts
            producer.produce(producer.golden_events("idem-new-ids", a.n))
            wait_for(lambda: store.count_events() >= 2 * a.n)
            replay_new = {"events": store.count_events(), "facts": store.count_facts(), **log_counts(logdir)}
        finally:
            stop_consumers(procs)
        result = {**meta, "n_conversations": a.n, "after_first_pass": first, "after_replay_same_event_ids": replay_same,
                  "after_replay_new_event_ids": replay_new,
                  "facts_unchanged": first["facts"] == replay_same["facts"] == replay_new["facts"]}

    else:  # failure
        logdir = OUT / "failure_logs"
        logdir.mkdir(parents=True, exist_ok=True)
        reset(store)
        procs = start_consumers(2, f"bench-fail-{int(time.time())}", logdir)
        killed = {}
        try:
            producer.produce(producer.golden_events("fail", a.n), raw=[("poison-1", b"{not json"), ("poison-2", b'{"event_id": 1}')])

            def maybe_kill():
                if not killed and store.count_events() >= a.kill_after:
                    procs[0][0].kill()  # hard kill (no graceful close, no final commit)
                    killed.update(at_processed=store.count_events(), at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            ok = wait_for(lambda: store.count_events() >= a.n, on_tick=maybe_kill)
            time.sleep(5)
        finally:
            stop_consumers(procs)
        dlq = count_topic(DLQ_TOPIC)
        m = metrics(store, a.n)
        result = {**meta, "completed": ok, "killed_consumer": "c0", "killed": killed, **m, "dlq_messages": dlq,
                  "log_counts": log_counts(logdir), "no_loss": m["processed_distinct"] == a.n}

    path = OUT / f"{a.mode}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


def count_topic(topic) -> int:
    from confluent_kafka import Consumer
    c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": f"count-{time.time()}", "auto.offset.reset": "earliest"})
    c.subscribe([topic])
    n, idle = 0, 0
    while idle < 5:
        m = c.poll(1.0)
        if m is None:
            idle += 1
        elif not m.error():
            n += 1
    c.close()
    return n


if __name__ == "__main__":
    main()
