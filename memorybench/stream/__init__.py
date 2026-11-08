"""v0.3 streaming ingestion: Kafka-API (Redpanda) producer/consumer + PostgreSQL/pgvector store."""
import os

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
TOPIC = os.getenv("KAFKA_TOPIC", "conversations")
DLQ_TOPIC = os.getenv("KAFKA_DLQ_TOPIC", "conversations.dlq")
GROUP = os.getenv("KAFKA_GROUP", "memorybench-extractors")
