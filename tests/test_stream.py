"""Streaming ingestion tests.

Unit tests (always run): idempotency + dead-letter logic of the Processor with in-memory fakes.
Integration tests (run when PG_DSN / KAFKA_BOOTSTRAP point to live services, e.g. the CI service
containers or `docker compose up -d redpanda postgres`): real PostgreSQL + pgvector SQL and a real
Kafka-API broker, with a fake extractor and fake embeddings (no LLM needed).
"""
import io
import json
import os
import uuid

import numpy as np
import pytest

from memorybench.logjson import get_logger, trace
from memorybench.schemas import Category, Fact
from memorybench.stream.processor import ConversationEvent, Message, Processor, encode

TURNS = [{"role": "customer", "text": "Please email me, I am on the Pro plan."}]


def fake_extractor(conv):
    text = " ".join(t.text for t in conv.turns if t.role == "customer").lower()
    facts = []
    if "email" in text:
        facts.append(Fact(category=Category.contact_preference, value="Prefers email"))
    if "pro plan" in text:
        facts.append(Fact(category=Category.plan, value="Has the Pro plan"))
    return facts, {"latency_ms": 1}


class FakeStore:
    def __init__(self):
        self.events, self.facts = {}, set()

    def is_processed(self, event_id):
        return event_id in self.events

    def add_event(self, event_id, customer_id, facts, meta=None):
        if event_id in self.events:
            return None
        self.events[event_id] = meta
        before = len(self.facts)
        self.facts |= {(customer_id, f.category.value, f.value) for f in facts}
        return len(self.facts) - before


def msg(ev, offset=0, trace_id="t1"):
    return Message(value=encode(ev), key=ev.customer_id.encode(), headers={"trace_id": trace_id, "produced_at_ms": "1700000000000"},
                   partition=0, offset=offset)


def make(store=None, extractor=fake_extractor):
    dlq = []
    log = get_logger("test-consumer", stream=io.StringIO())
    return Processor(store or FakeStore(), extractor, lambda m, err: dlq.append((m, err)), backoff_s=0, log=log), dlq


def test_redelivery_is_idempotent():
    p, dlq = make()
    ev = ConversationEvent(event_id="e1", conv_id="c1", customer_id="cust_1", turns=TURNS)
    assert p.handle(msg(ev)) == "processed"
    assert p.handle(msg(ev, offset=0)) == "duplicate"
    assert len(p.store.facts) == 2 and len(p.store.events) == 1 and dlq == []


def test_new_event_with_same_facts_does_not_duplicate_facts():
    p, _ = make()
    for i in range(3):
        p.handle(msg(ConversationEvent(event_id=f"e{i}", conv_id="c", customer_id="cust_1", turns=TURNS), offset=i))
    assert len(p.store.events) == 3 and len(p.store.facts) == 2


@pytest.mark.parametrize("payload", [b"not json", b'{"event_id": "x"}', None])
def test_malformed_goes_to_dlq(payload):
    p, dlq = make()
    assert p.handle(Message(value=payload, partition=1, offset=7)) == "dlq"
    assert len(dlq) == 1 and p.store.events == {}


def test_transient_extract_failure_raises_without_writing():
    calls = {"n": 0}

    def flaky(conv):
        calls["n"] += 1
        raise ConnectionError("ollama down")

    p, dlq = make(extractor=flaky)
    with pytest.raises(ConnectionError):
        p.handle(msg(ConversationEvent(event_id="e1", conv_id="c", customer_id="cust_1", turns=TURNS)))
    assert calls["n"] == 3 and p.store.events == {} and dlq == []  # offset not committed -> redelivered later


def test_extract_retry_then_success():
    state = {"n": 0}

    def flaky_once(conv):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("slow")
        return fake_extractor(conv)

    p, _ = make(extractor=flaky_once)
    assert p.handle(msg(ConversationEvent(event_id="e1", conv_id="c", customer_id="cust_1", turns=TURNS))) == "processed"


def test_json_log_carries_trace_id():
    buf = io.StringIO()
    log = get_logger("test-json", stream=buf)
    with trace("abc123"):
        log.info("hello", extra={"fields": {"k": 1}})
    rec = json.loads(buf.getvalue())
    assert rec["trace_id"] == "abc123" and rec["k"] == 1 and rec["msg"] == "hello"


# ---------------------------------------------------------------- integration (live services)

PG_DSN = os.getenv("PG_DSN")
KAFKA = os.getenv("KAFKA_BOOTSTRAP")


def fake_embed(texts):
    vocab = ["email", "phone", "pro", "plan", "invoice", "pdf"]
    v = np.zeros((len(texts), 768), dtype=np.float32)
    for i, t in enumerate(texts):
        v[i, :len(vocab)] = [t.lower().count(w) for w in vocab]
        v[i, 767] = 1.0
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def pg():
    if not PG_DSN:
        pytest.skip("PG_DSN not set")
    from memorybench.pgstore import PostgresFactStore
    s = PostgresFactStore(PG_DSN, embed=fake_embed)
    s.reset()
    yield s
    s.reset()
    s.close()


@pytest.mark.integration
def test_pg_store_add_list_recall(pg):
    facts = [Fact(category=Category.plan, value="Has the Pro plan"),
             Fact(category=Category.preference, value="Wants invoices as PDF"),
             Fact(category=Category.contact_preference, value="Prefers email over phone")]
    assert pg.add("c1", facts) == 3
    assert pg.add("c1", facts) == 0  # UNIQUE(customer_id, category, value)
    assert [f.value for f in pg.list_facts("c1")] == [f.value for f in facts]
    hits = pg.recall("c1", "invoice pdf", k=2)
    assert hits[0].fact.value == "Wants invoices as PDF" and len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert pg.recall("nobody", "x") == []


@pytest.mark.integration
def test_pg_add_event_idempotent(pg):
    f = [Fact(category=Category.plan, value="Has the Pro plan")]
    assert pg.add_event("e1", "c1", f, {"consumer_id": "t"}) == 1
    assert pg.add_event("e1", "c1", f) is None
    assert pg.add_event("e2", "c1", f) == 0
    assert pg.count_facts() == 1 and pg.count_events() == 2


@pytest.mark.integration
def test_kafka_to_postgres_end_to_end(pg):
    if not KAFKA:
        pytest.skip("KAFKA_BOOTSTRAP not set")
    from confluent_kafka import Consumer

    from memorybench.stream import DLQ_TOPIC, consumer, producer
    run = uuid.uuid4().hex[:8]
    topic = f"it-conversations-{run}"
    producer.ensure_topics(KAFKA, partitions=3, topic=topic)
    evs = [ConversationEvent(event_id=f"{run}:{i}", conv_id=f"c{i}", customer_id=f"cust_{i % 3}", turns=TURNS) for i in range(6)]
    producer.produce(evs, KAFKA, raw=[("bad", b"{broken")], topic=topic)
    producer.produce(evs[:2], KAFKA, topic=topic)  # replay two events
    stats = consumer.run(pg, fake_extractor, "it", bootstrap=KAFKA, group=f"it-{run}", max_messages=9, idle_exit_s=20, topic=topic)
    assert stats == {"processed": 6, "duplicate": 2, "dlq": 1}
    assert pg.count_events() == 6 and pg.count_facts() == 6  # 3 customers x 2 facts
    c = Consumer({"bootstrap.servers": KAFKA, "group.id": f"dlq-{run}", "auto.offset.reset": "earliest"})
    c.subscribe([DLQ_TOPIC])
    found = False
    for _ in range(30):
        m = c.poll(1.0)
        if m is not None and not m.error() and m.value() == b"{broken" and m.headers() and dict(m.headers()).get("source_offset") is not None:
            found = True
            break
    c.close()
    assert found
