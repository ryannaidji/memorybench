"""Publish conversation events to Kafka/Redpanda, keyed by customer_id (per-customer ordering).

    python -m memorybench.stream.producer --source golden --limit 30
    python -m memorybench.stream.producer --source load --n 200 --run r1
"""
import argparse
import json
import time
from pathlib import Path

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from . import BOOTSTRAP, DLQ_TOPIC, TOPIC
from ..logjson import get_logger, new_trace_id, trace
from .processor import ConversationEvent, encode

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden.jsonl"
log = get_logger("producer")


def golden_events(run: str, limit: int | None = None) -> list[ConversationEvent]:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [ConversationEvent(event_id=f"{run}:{r['conv_id']}", conv_id=r["conv_id"], customer_id=r["customer_id"],
                              turns=r["turns"]) for r in rows[:limit]]


def load_events(run: str, n: int) -> list[ConversationEvent]:
    """Load generator: cycles the golden conversations under fresh customer ids (cust_07 -> cust_07_l2)."""
    base = golden_events("x")
    out = []
    for i in range(n):
        b, rep = base[i % len(base)], i // len(base)
        cid = b.customer_id if rep == 0 else f"{b.customer_id}_l{rep}"
        out.append(ConversationEvent(event_id=f"{run}:load{i}", conv_id=f"{b.conv_id}_l{rep}", customer_id=cid, turns=b.turns))
    return out


def ensure_topics(bootstrap: str = BOOTSTRAP, partitions: int = 3, recreate: bool = False, topic: str = TOPIC):
    admin = AdminClient({"bootstrap.servers": bootstrap})
    if recreate:
        existing = set(admin.list_topics(timeout=10).topics)
        dels = [t for t in (topic, DLQ_TOPIC) if t in existing]
        for f in admin.delete_topics(dels).values() if dels else []:
            f.result()
        time.sleep(2)
    existing = set(admin.list_topics(timeout=10).topics)
    new = [NewTopic(topic, partitions, 1), NewTopic(DLQ_TOPIC, 1, 1)]
    futs = admin.create_topics([t for t in new if t.topic not in existing])
    for f in futs.values():
        f.result()


def produce(events: list[ConversationEvent], bootstrap: str = BOOTSTRAP, raw: list[tuple[str, bytes]] = (), topic: str = TOPIC) -> int:
    p = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all", "linger.ms": 5})
    sent = 0
    for ev in events:
        tid = new_trace_id()
        with trace(tid):
            p.produce(topic, key=ev.customer_id.encode(), value=encode(ev),
                      headers={"trace_id": tid, "produced_at_ms": str(int(time.time() * 1000)), "event_id": ev.event_id})
            log.info("produced", extra={"fields": {"event_id": ev.event_id, "customer_id": ev.customer_id}})
        sent += 1
        p.poll(0)
    for key, value in raw:  # malformed payloads, used to exercise the dead-letter path
        p.produce(topic, key=key.encode(), value=value, headers={"trace_id": new_trace_id(),
                                                                 "produced_at_ms": str(int(time.time() * 1000))})
    if p.flush(30) != 0:
        raise RuntimeError("producer flush timed out")
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["golden", "load"], default="golden")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--run", default="r1", help="run id; same run id => same event ids (replay)")
    ap.add_argument("--partitions", type=int, default=3)
    a = ap.parse_args()
    ensure_topics(partitions=a.partitions)
    evs = golden_events(a.run, a.limit) if a.source == "golden" else load_events(a.run, a.n)
    print(json.dumps({"produced": produce(evs)}))


if __name__ == "__main__":
    main()
