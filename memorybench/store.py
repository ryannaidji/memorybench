"""SQLite fact store with embedding search (cosine similarity in NumPy).

Kept dependency-free on purpose: one file, no server. The interface (add / list / recall)
is what a Postgres + pgvector backend would implement.
"""
import sqlite3
from datetime import datetime, timezone

import numpy as np

from . import llm
from .schemas import Fact, RecallHit, StoredFact


class FactStore:
    def __init__(self, path: str = ":memory:", embed=llm.embed):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("""CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY, customer_id TEXT NOT NULL, category TEXT NOT NULL,
            value TEXT NOT NULL, created_at TEXT NOT NULL, embedding BLOB NOT NULL,
            UNIQUE(customer_id, category, value))""")
        self.embed = embed

    def add(self, customer_id: str, facts: list[Fact]) -> int:
        if not facts:
            return 0
        vecs = self.embed([f"{f.category.value}: {f.value}" for f in facts])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        added = 0
        with self.con:
            for f, v in zip(facts, vecs):
                cur = self.con.execute(
                    "INSERT OR IGNORE INTO facts (customer_id, category, value, created_at, embedding) VALUES (?,?,?,?,?)",
                    (customer_id, f.category.value, f.value, now, v.astype(np.float32).tobytes()))
                added += cur.rowcount
        return added

    def list_facts(self, customer_id: str) -> list[StoredFact]:
        rows = self.con.execute("SELECT id, customer_id, category, value, created_at FROM facts WHERE customer_id=? ORDER BY id",
                                (customer_id,)).fetchall()
        return [StoredFact(id=r[0], customer_id=r[1], category=r[2], value=r[3], created_at=r[4]) for r in rows]

    def recall(self, customer_id: str, query: str, k: int = 5) -> list[RecallHit]:
        rows = self.con.execute("SELECT id, customer_id, category, value, created_at, embedding FROM facts WHERE customer_id=?",
                                (customer_id,)).fetchall()
        if not rows:
            return []
        q = self.embed([query])[0]
        mat = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
        scores = mat @ q
        order = np.argsort(-scores)[:k]
        return [RecallHit(fact=StoredFact(id=rows[i][0], customer_id=rows[i][1], category=rows[i][2], value=rows[i][3],
                                          created_at=rows[i][4]), score=round(float(scores[i]), 4)) for i in order]


def get_store(**kw):
    """STORE=sqlite (default) or STORE=postgres (PostgresFactStore, PG_DSN)."""
    import os
    if os.getenv("STORE", "sqlite").lower() == "postgres":
        from .pgstore import PostgresFactStore
        return PostgresFactStore(**kw)
    return FactStore(os.getenv("MEMORY_DB", "memory.db"), **kw)
