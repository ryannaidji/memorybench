"""Broker-independent message handling: parse -> (skip if seen) -> extract -> idempotent write.

Kept free of Kafka imports so it can be unit-tested with fakes. The consumer commits the
offset only after `handle` returns, which gives at-least-once delivery; the processed_events
ledger + UNIQUE fact constraint turn redeliveries into no-ops (effectively-once side effects).
"""
import json
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from ..logjson import get_logger, trace
from ..schemas import Conversation, Fact, Turn


class ConversationEvent(BaseModel):
    event_id: str
    conv_id: str
    customer_id: str
    turns: list[Turn]


def encode(ev: ConversationEvent) -> bytes:
    return ev.model_dump_json().encode()


def decode(value: bytes | None) -> ConversationEvent:
    if value is None:
        raise ValueError("empty message")
    return ConversationEvent.model_validate_json(value)


def headers_dict(headers) -> dict[str, str]:
    return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in (headers or [])}


@dataclass
class Message:
    """Minimal view of a Kafka message (confluent_kafka.Message is adapted to this)."""
    value: bytes | None
    key: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    partition: int | None = None
    offset: int | None = None


class PoisonMessage(Exception):
    pass


class Processor:
    def __init__(self, store, extractor, send_dlq, consumer_id: str = "c0", parse_attempts: int = 3,
                 extract_attempts: int = 3, backoff_s: float = 0.5, log=None):
        self.store, self.extractor, self.send_dlq = store, extractor, send_dlq
        self.consumer_id, self.parse_attempts, self.extract_attempts, self.backoff_s = consumer_id, parse_attempts, extract_attempts, backoff_s
        self.log = log or get_logger("consumer")

    def _parse(self, msg: Message) -> ConversationEvent:
        err = None
        for _ in range(self.parse_attempts):
            try:
                return decode(msg.value)
            except (ValidationError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
                err = e
        raise PoisonMessage(str(err)[:300])

    def _extract(self, ev: ConversationEvent) -> tuple[list[Fact], dict]:
        conv = Conversation(customer_id=ev.customer_id, turns=ev.turns)
        for attempt in range(1, self.extract_attempts + 1):
            try:
                return self.extractor(conv)
            except Exception as e:  # transient (LLM server down, timeout): retry, then let the consumer crash
                self.log.warning("extract_failed", extra={"fields": {"attempt": attempt, "error": str(e)[:200]}})
                if attempt == self.extract_attempts:
                    raise
                time.sleep(self.backoff_s * attempt)

    def handle(self, msg: Message) -> str:
        """Returns 'processed', 'duplicate' or 'dlq'. Raises on transient failure (offset must NOT be committed)."""
        h = msg.headers or {}
        with trace(h.get("trace_id")):
            try:
                ev = self._parse(msg)
            except PoisonMessage as e:
                self.send_dlq(msg, str(e))
                self.log.error("sent_to_dlq", extra={"fields": {"partition": msg.partition, "offset": msg.offset, "error": str(e)[:200]}})
                return "dlq"
            ctx = {"event_id": ev.event_id, "customer_id": ev.customer_id, "partition": msg.partition, "offset": msg.offset}
            if self.store.is_processed(ev.event_id):  # cheap pre-check: skip the LLM on redelivery
                self.log.info("duplicate_skipped", extra={"fields": ctx})
                return "duplicate"
            facts, info = self._extract(ev)
            produced_at = h.get("produced_at_ms")
            meta = {"conv_id": ev.conv_id, "trace_id": h.get("trace_id"), "consumer_id": self.consumer_id,
                    "partition": msg.partition, "offset": msg.offset, "extract_ms": info.get("latency_ms"),
                    "produced_at": _ts(produced_at) if produced_at else None}
            added = self.store.add_event(ev.event_id, ev.customer_id, facts, meta)
            if added is None:  # another consumer won the race (e.g. during a rebalance)
                self.log.info("duplicate_skipped", extra={"fields": ctx})
                return "duplicate"
            self.log.info("processed", extra={"fields": {**ctx, "facts": len(facts), "facts_added": added,
                                                         "extract_ms": info.get("latency_ms")}})
            return "processed"


def _ts(ms: str):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc)
