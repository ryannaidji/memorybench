"""Unit tests: no LLM server needed (LLM and embeddings are faked)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from memorybench import api, extract as ex, llm
from memorybench.schemas import Category, Conversation, Fact
from memorybench.store import FactStore

VOCAB = ["email", "phone", "pro", "plan", "crash", "invoice", "pdf", "spanish", "admin", "clinic"]


def fake_embed(texts):
    v = np.array([[t.lower().count(w) for w in VOCAB] + [1.0] for t in texts], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def store():
    return FactStore(":memory:", embed=fake_embed)


def test_store_dedupes_and_lists(store):
    facts = [Fact(category=Category.contact_preference, value="Prefers email over phone"),
             Fact(category=Category.contact_preference, value="Prefers email over phone")]
    assert store.add("c1", facts) == 1
    assert store.add("c1", facts[:1]) == 0
    assert [f.value for f in store.list_facts("c1")] == ["Prefers email over phone"]
    assert store.list_facts("other") == []


def test_recall_ranks_relevant_fact_first(store):
    store.add("c1", [Fact(category=Category.plan, value="Has the Pro plan"),
                     Fact(category=Category.preference, value="Wants invoices as PDF"),
                     Fact(category=Category.issue, value="App crash on reports tab")])
    hits = store.recall("c1", "invoice pdf format", k=2)
    assert hits[0].fact.value == "Wants invoices as PDF"
    assert len(hits) == 2


def test_extract_parses_dedupes_and_strips(monkeypatch):
    payload = {"facts": [{"category": "plan", "value": " Has the Pro plan "},
                         {"category": "plan", "value": "has the pro plan"},
                         {"category": "issue", "value": ""},
                         {"category": "plan", "value": "Not specified"},
                         {"category": "profile", "value": "unknown"}]}
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: json.dumps(payload))
    conv = Conversation(customer_id="c1", turns=[{"role": "customer", "text": "We're on Pro."}])
    facts, info = ex.extract(conv)
    assert [(f.category.value, f.value) for f in facts] == [("plan", "Has the Pro plan")]
    assert info["parse_error"] is None


def test_extract_handles_invalid_json(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: "not json")
    facts, info = ex.extract(Conversation(customer_id="c", turns=[{"role": "customer", "text": "hi"}]))
    assert facts == [] and info["parse_error"]


def test_api_roundtrip(monkeypatch):
    monkeypatch.setattr(api, "store", FactStore(":memory:", embed=fake_embed))
    monkeypatch.setattr(llm, "chat_json",
                        lambda *a, **k: json.dumps({"facts": [{"category": "contact_preference", "value": "Prefers email"}]}))
    c = TestClient(api.app)
    r = c.post("/conversations", json={"customer_id": "c9", "turns": [{"role": "customer", "text": "email me"}]})
    assert r.status_code == 200 and r.json()["added"] == 1
    assert c.get("/customers/c9/facts").json()[0]["value"] == "Prefers email"
    assert c.get("/customers/c9/recall", params={"q": "email"}).json()[0]["fact"]["value"] == "Prefers email"
