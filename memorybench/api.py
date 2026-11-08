"""FastAPI service: POST a conversation, get durable facts; recall them later.

Run: uvicorn memorybench.api:app --reload
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from .extract import extract
from .schemas import Conversation, Fact, RecallHit, StoredFact
from .store import get_store

app = FastAPI(title="MemoryBench", version="0.1.0")
store = get_store()


class IngestResponse(BaseModel):
    facts: list[Fact]
    added: int
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/conversations", response_model=IngestResponse)
def ingest(conv: Conversation):
    facts, info = extract(conv)
    added = store.add(conv.customer_id, facts)
    return IngestResponse(facts=facts, added=added, latency_ms=info["latency_ms"])


@app.get("/customers/{customer_id}/facts", response_model=list[StoredFact])
def facts(customer_id: str):
    return store.list_facts(customer_id)


@app.get("/customers/{customer_id}/recall", response_model=list[RecallHit])
def recall(customer_id: str, q: str, k: int = 5):
    return store.recall(customer_id, q, k)
