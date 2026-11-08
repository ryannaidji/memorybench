"""Provider-agnostic LLM + embedding client.

Uses the OpenAI-compatible API, so the same code runs against a local Ollama server
(default, free) or any OpenAI-compatible endpoint by changing environment variables:

    LLM_BASE_URL   default http://localhost:11434/v1
    LLM_MODEL      default qwen2.5:3b
    EMBED_MODEL    default nomic-embed-text
    LLM_API_KEY    default "ollama" (ignored by Ollama)
"""
import os
from functools import lru_cache

import numpy as np
from openai import OpenAI


def settings():
    return {
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "model": os.getenv("LLM_MODEL", "qwen2.5:3b"),
        "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
        "api_key": os.getenv("LLM_API_KEY", "ollama"),
    }


@lru_cache(maxsize=1)
def client() -> OpenAI:
    s = settings()
    return OpenAI(base_url=s["base_url"], api_key=s["api_key"], timeout=300)


def chat_json(system: str, user: str, schema: dict, model: str | None = None) -> str:
    """Return the raw JSON string produced under a JSON-schema constrained decode."""
    resp = client().chat.completions.create(
        model=model or settings()["model"],
        temperature=0,
        max_tokens=400,  # guards against runaway whitespace under constrained decoding
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "facts", "schema": schema}},
    )
    return resp.choices[0].message.content or "{}"


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, shape (n, d)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    resp = client().embeddings.create(model=settings()["embed_model"], input=texts)
    v = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)
