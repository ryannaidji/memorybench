-- 001: facts with pgvector embeddings + processed-event ledger (idempotency).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS facts (
    id          BIGSERIAL PRIMARY KEY,
    customer_id TEXT        NOT NULL,
    category    TEXT        NOT NULL,
    value       TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(768) NOT NULL,
    CONSTRAINT facts_customer_category_value_key UNIQUE (customer_id, category, value)
);
CREATE INDEX IF NOT EXISTS facts_customer_idx ON facts (customer_id);

-- One row per consumed event. The PRIMARY KEY makes redelivery a no-op: a message
-- processed before a crash (DB committed, offset not yet committed) is skipped on replay.
CREATE TABLE IF NOT EXISTS processed_events (
    event_id        TEXT PRIMARY KEY,
    conv_id         TEXT,
    customer_id     TEXT        NOT NULL,
    trace_id        TEXT,
    consumer_id     TEXT,
    kafka_partition INT,
    kafka_offset    BIGINT,
    produced_at     TIMESTAMPTZ,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    facts_added     INT         NOT NULL DEFAULT 0,
    extract_ms      INT
);
