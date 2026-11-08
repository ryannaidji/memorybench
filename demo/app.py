"""MemoryBench demo: paste a support conversation, see the facts extracted, ask the memory.

Run locally (needs Ollama): streamlit run demo/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memorybench.extract import extract  # noqa: E402
from memorybench.schemas import Conversation  # noqa: E402
from memorybench.store import FactStore  # noqa: E402

st.set_page_config(page_title="MemoryBench", page_icon=":material/memory:", layout="wide")
st.title("MemoryBench", anchor=False)
st.caption("Durable customer memory from support conversations · local open-weight models via Ollama")


@st.cache_resource
def get_store():
    return FactStore(":memory:")


store = get_store()
golden = [json.loads(l) for l in (Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl").read_text().splitlines()]
sample = "\n".join(f"{t['role']}: {t['text']}" for t in golden[0]["turns"])

left, right = st.columns(2, gap="large")
with left, st.container(border=True):
    st.subheader("1. Ingest a conversation", anchor=False)
    cid = st.text_input("Customer ID", value="demo_customer")
    text = st.text_area("Conversation (one `role: text` per line)", value=sample, height=300)
    if st.button("Extract facts", type="primary", icon=":material/auto_awesome:"):
        turns = []
        for line in text.splitlines():
            role, _, msg = line.partition(":")
            if msg.strip():
                turns.append({"role": role.strip().lower(), "text": msg.strip()})
        with st.spinner("Extracting…"):
            facts, info = extract(Conversation(customer_id=cid, turns=turns))
            added = store.add(cid, facts)
        st.success(f"{len(facts)} facts extracted, {added} new · {info['latency_ms']} ms")
        st.dataframe([{"category": f.category.value, "fact": f.value} for f in facts], hide_index=True,
                     alt="Extracted facts")

with right, st.container(border=True):
    st.subheader("2. Ask the memory", anchor=False)
    q = st.text_input("Question", value="How should we contact this customer?")
    if st.button("Recall", icon=":material/search:"):
        hits = store.recall(cid, q, k=5)
        if not hits:
            st.info("No memory for this customer yet.")
        st.dataframe([{"score": h.score, "category": h.fact.category, "fact": h.fact.value} for h in hits],
                     hide_index=True, alt="Recalled facts ranked by similarity")
    st.divider()
    st.markdown("**All facts for this customer**")
    st.dataframe([{"category": f.category, "fact": f.value, "added": f.created_at} for f in store.list_facts(cid)],
                 hide_index=True, alt="All stored facts")
