"""Consumer-group worker: run 1..K of these; Kafka assigns partitions among them.

    STORE=postgres python -m memorybench.stream.consumer --id c1

Offsets are committed manually, synchronously, only after the Postgres transaction commits
(at-least-once). Malformed messages go to the dead-letter topic, then their offset is committed.
"""
import argparse
import os
import signal
import time

from confluent_kafka import Consumer, KafkaError, Producer

from . import BOOTSTRAP, DLQ_TOPIC, GROUP, TOPIC
from ..logjson import get_logger
from .processor import Message, Processor, headers_dict

log = get_logger("consumer")


def make_dlq_sender(producer: Producer):
    def send(msg: Message, error: str):
        hdrs = dict(msg.headers or {})
        hdrs.update({"dlq_error": error[:500], "source_partition": str(msg.partition), "source_offset": str(msg.offset)})
        producer.produce(DLQ_TOPIC, key=msg.key, value=msg.value, headers=hdrs)
        if producer.flush(10) != 0:
            raise RuntimeError("DLQ publish failed")  # do not commit the offset if the DLQ write is not durable
    return send


def run(store, extractor, consumer_id: str, bootstrap: str = BOOTSTRAP, group: str = GROUP,
        max_messages: int | None = None, idle_exit_s: float | None = None, topic: str = TOPIC) -> dict:
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": group, "enable.auto.commit": False,
                  "auto.offset.reset": "earliest", "client.id": consumer_id, "session.timeout.ms": 10000,
                  "max.poll.interval.ms": 600000})
    dlq = Producer({"bootstrap.servers": bootstrap, "acks": "all"})
    proc = Processor(store, extractor, make_dlq_sender(dlq), consumer_id=consumer_id, log=log)
    stats = {"processed": 0, "duplicate": 0, "dlq": 0}
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    c.subscribe([topic], on_assign=lambda _c, parts: log.info("assigned", extra={"fields": {
        "consumer_id": consumer_id, "partitions": [p.partition for p in parts]}}))
    log.info("consumer_started", extra={"fields": {"consumer_id": consumer_id, "group": group, "topic": topic}})
    last = time.time()
    try:
        while not stop["flag"]:
            m = c.poll(1.0)
            if m is None:
                if idle_exit_s and time.time() - last > idle_exit_s:
                    break
                continue
            if m.error():
                if m.error().code() != KafkaError._PARTITION_EOF:
                    log.error("kafka_error", extra={"fields": {"error": str(m.error())}})
                continue
            last = time.time()
            outcome = proc.handle(Message(value=m.value(), key=m.key(), headers=headers_dict(m.headers()),
                                          partition=m.partition(), offset=m.offset()))
            c.commit(message=m, asynchronous=False)  # commit AFTER the DB write
            stats[outcome] += 1
            if max_messages and sum(stats.values()) >= max_messages:
                break
    finally:
        c.close()
        log.info("consumer_stopped", extra={"fields": {"consumer_id": consumer_id, **stats}})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=f"c-{os.getpid()}")
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--idle-exit", type=float, default=None)
    a = ap.parse_args()
    os.environ.setdefault("STORE", "postgres")
    from ..extract import extract
    from ..store import get_store
    run(get_store(), extract, a.id, group=a.group, idle_exit_s=a.idle_exit)


if __name__ == "__main__":
    main()
