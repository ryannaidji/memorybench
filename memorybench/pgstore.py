"""PostgreSQL + pgvector fact store. Same interface as the SQLite FactStore (add / list_facts / recall),
plus `add_event`, which writes an event's facts and its ledger row in ONE transaction (idempotent).

Select it with STORE=postgres and PG_DSN=postgresql://memorybench:memorybench@localhost:55432/memorybench
"""
import os
from datetime import datetime
from pathlib import Path

import psycopg

from . import llm
from .schemas import Fact, RecallHit, StoredFact

MIGRATIONS = Path(__file__).parent / "migrations"
DEFAULT_DSN = "postgresql://memorybench:memorybench@localhost:55432/memorybench"

INSERT_FACT = """INSERT INTO facts (customer_id, category, value, embedding) VALUES (%s, %s, %s, %s::vector)
                 ON CONFLICT ON CONSTRAINT facts_customer_category_value_key DO NOTHING"""
RECALL = """SELECT id, customer_id, category, value, created_at, 1 - (embedding <=> %s::vector) AS score
            FROM facts WHERE customer_id = %s ORDER BY embedding <=> %s::vector LIMIT %s"""


def vec_literal(v) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


def migrate(con: psycopg.Connection) -> list[str]:
    """Apply migrations/*.sql in order, once each (tracked in schema_migrations)."""
    applied = []
    with con.transaction():
        con.execute("SELECT pg_advisory_xact_lock(4242)")  # several consumers may start at once
        con.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())")
        done = {r[0] for r in con.execute("SELECT name FROM schema_migrations")}
        for f in sorted(MIGRATIONS.glob("*.sql")):
            if f.name not in done:
                con.execute(f.read_text(encoding="utf-8"))
                con.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))
                applied.append(f.name)
    return applied


class PostgresFactStore:
    def __init__(self, dsn: str | None = None, embed=llm.embed):
        self.con = psycopg.connect(dsn or os.getenv("PG_DSN", DEFAULT_DSN), autocommit=True)
        migrate(self.con)
        self.embed = embed

    def _rows(self, facts: list[Fact]):
        vecs = self.embed([f"{f.category.value}: {f.value}" for f in facts]) if facts else []
        return [(f, vec_literal(v)) for f, v in zip(facts, vecs)]

    def _insert(self, customer_id: str, rows) -> int:
        added = 0
        for f, v in rows:
            added += self.con.execute(INSERT_FACT, (customer_id, f.category.value, f.value, v)).rowcount
        return added

    def add(self, customer_id: str, facts: list[Fact]) -> int:
        rows = self._rows(facts)
        with self.con.transaction():
            return self._insert(customer_id, rows)

    def is_processed(self, event_id: str) -> bool:
        return self.con.execute("SELECT 1 FROM processed_events WHERE event_id = %s", (event_id,)).fetchone() is not None

    def add_event(self, event_id: str, customer_id: str, facts: list[Fact], meta: dict | None = None) -> int | None:
        """Write facts + ledger row atomically. Returns facts added, or None if the event was already processed."""
        meta = meta or {}
        rows = self._rows(facts)  # embed outside the transaction
        with self.con.transaction():
            cur = self.con.execute(
                """INSERT INTO processed_events (event_id, conv_id, customer_id, trace_id, consumer_id, kafka_partition,
                       kafka_offset, produced_at, extract_ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (event_id) DO NOTHING""",
                (event_id, meta.get("conv_id"), customer_id, meta.get("trace_id"), meta.get("consumer_id"),
                 meta.get("partition"), meta.get("offset"), meta.get("produced_at"), meta.get("extract_ms")))
            if cur.rowcount == 0:
                return None
            added = self._insert(customer_id, rows)
            self.con.execute("UPDATE processed_events SET facts_added = %s WHERE event_id = %s", (added, event_id))
        return added

    def list_facts(self, customer_id: str) -> list[StoredFact]:
        rows = self.con.execute("SELECT id, customer_id, category, value, created_at FROM facts WHERE customer_id = %s ORDER BY id",
                                (customer_id,)).fetchall()
        return [StoredFact(id=r[0], customer_id=r[1], category=r[2], value=r[3], created_at=_iso(r[4])) for r in rows]

    def recall(self, customer_id: str, query: str, k: int = 5) -> list[RecallHit]:
        q = vec_literal(self.embed([query])[0])
        rows = self.con.execute(RECALL, (q, customer_id, q, k)).fetchall()
        return [RecallHit(fact=StoredFact(id=r[0], customer_id=r[1], category=r[2], value=r[3], created_at=_iso(r[4])),
                          score=round(float(r[5]), 4)) for r in rows]

    def count_facts(self) -> int:
        return self.con.execute("SELECT count(*) FROM facts").fetchone()[0]

    def count_events(self) -> int:
        return self.con.execute("SELECT count(*) FROM processed_events").fetchone()[0]

    def reset(self):
        self.con.execute("TRUNCATE facts, processed_events RESTART IDENTITY")

    def close(self):
        self.con.close()


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")
