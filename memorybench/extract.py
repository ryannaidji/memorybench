"""Turn a support conversation into durable customer facts."""
import json
import os
import re
import time

from pydantic import ValidationError

from . import llm
from .schemas import Conversation, Fact, FactList

PROMPT_V1 = """You extract durable customer memory from customer-support conversations.
Return ONLY facts about the CUSTOMER that will still be useful in a future conversation:
- contact_preference: how or when they want to be contacted (channel, time window)
- plan: the subscription plan or product tier they have or switched to
- issue: a concrete problem they reported (what failed, when)
- preference: stable preferences (language, invoice format, product preferences)
- profile: stable facts about them or their business (role, team size, location, industry)
Rules:
- Each value is one short, self-contained statement in English (max 15 words), e.g. "Prefers email over phone".
- Do not include facts about the agent, greetings, small talk or things already resolved and irrelevant.
- Do not guess. If nothing durable was said, return an empty list.
- Never repeat the same fact twice."""

PROMPT_V2 = """You extract durable customer memory from a customer-support conversation.
Only use sentences said by the CUSTOMER. Ignore everything the agent says.

Categories (pick exactly one per fact):
- contact_preference: HOW or WHEN to contact them (email, phone, SMS, WhatsApp, time window). Language is NOT a contact preference.
- plan: their subscription plan, tier, seats or billing cycle.
- issue: a concrete problem they report (what is broken or wrong).
- preference: language, reply format, invoice format, UI theme or other stable preferences.
- profile: who they are or their organisation (role, team size, business type, location).

Strict rules:
1. Extract a fact ONLY if the customer explicitly stated it. Never infer, guess or add details (no industry, size or location unless said).
2. One customer sentence gives at most ONE fact. Do not split it into several facts.
3. Small talk, apologies, thanks, greetings, goodbyes, weather, typos: NOT facts.
4. Write each value as a short third-person statement, max 12 words, keeping the customer's key details (numbers, names, times).
5. If the customer said nothing durable, return {"facts": []}.

Example
CUSTOMER: Sorry, typing from my phone.
CUSTOMER: We have the Team plan, 10 seats. Please only call me in the morning.
CUSTOMER: Also, I'd like replies in French.
Output: {"facts": [
 {"category": "plan", "value": "Has the Team plan with 10 seats"},
 {"category": "contact_preference", "value": "Wants calls only in the morning"},
 {"category": "preference", "value": "Prefers replies in French"}]}"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
PLACEHOLDER = re.compile(r"^(not (specified|mentioned|provided|stated)|unknown|none|n/?a|null|no (info|information)|-+)\.?$", re.I)
SYSTEM_PROMPT = PROMPTS[os.getenv("PROMPT_VERSION", "v2")]


def render(conv: Conversation) -> str:
    return "\n".join(f"{t.role.upper()}: {t.text}" for t in conv.turns)


def extract(conv: Conversation, model: str | None = None, retries: int = 1, prompt: str | None = None) -> tuple[list[Fact], dict]:
    """Return (facts, info) where info has latency_ms and parse status."""
    schema = FactList.model_json_schema()
    user = f"Conversation:\n{render(conv)}\n\nReturn the facts as JSON."
    t0 = time.perf_counter()
    raw, err = "", None
    for _ in range(retries + 1):
        raw = llm.chat_json(PROMPTS.get(prompt, SYSTEM_PROMPT) if prompt else SYSTEM_PROMPT, user, schema, model=model)
        try:
            facts = FactList.model_validate_json(raw).facts
            break
        except (ValidationError, json.JSONDecodeError) as e:
            err, facts = str(e)[:200], []
    seen, unique = set(), []
    for f in facts:
        key = (f.category, f.value.strip().lower())
        if f.value.strip() and not PLACEHOLDER.match(f.value.strip()) and key not in seen:
            seen.add(key)
            unique.append(Fact(category=f.category, value=f.value.strip()))
    info = {"latency_ms": round((time.perf_counter() - t0) * 1000), "parse_error": err if not facts and err else None}
    return unique, info
