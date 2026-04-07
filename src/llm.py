"""
llm.py — LLM factory backed by OpenRouter.

All agents use ChatOpenAI pointed at the OpenRouter base URL.
This keeps all model routing in one place and makes model swaps trivial.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


@lru_cache(maxsize=64)
def get_llm(model: str, temperature: float = 0.0, api_key: str = "") -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance pointing at OpenRouter.

    api_key is the per-request key supplied by the UI.  Falls back to the
    OPENROUTER_API_KEY env var when the caller supplies an empty string.
    The lru_cache key includes api_key so different callers get isolated
    instances without sharing one another's credentials.
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise RuntimeError(
            "No OpenRouter API key provided. "
            "Enter your key in the UI or set OPENROUTER_API_KEY in .env."
        )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=key,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/research-agent",
            "X-Title":      "Production Research Agent",
        },
    )
