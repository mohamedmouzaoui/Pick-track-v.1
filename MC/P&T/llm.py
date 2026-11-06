"""Tiny provider-agnostic LLM client. Falls back to 'offline' (rules only) with no API key."""
from __future__ import annotations

import os
from typing import Optional

import requests

PROVIDERS = ("groq", "anthropic", "offline")


class LLMClient:
    def __init__(self, provider: str = "offline", api_key: str | None = None,
                 model: str | None = None, timeout: float = 20.0):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        self.provider, self.api_key, self.timeout = provider, api_key, timeout
        self.model = model or {
            "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            "offline": "rules",
        }[provider]
        self.last_error: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMClient":
        forced = os.getenv("LLM_PROVIDER", "").lower()
        groq, anth = os.getenv("GROQ_API_KEY"), os.getenv("ANTHROPIC_API_KEY")
        if forced == "offline":
            return cls("offline")
        if forced == "groq" or (not forced and groq):
            return cls("groq", groq) if groq else cls("offline")
        if forced == "anthropic" or (not forced and anth):
            return cls("anthropic", anth) if anth else cls("offline")
        return cls("offline")

    @property
    def enabled(self) -> bool:
        return self.provider != "offline" and bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 400) -> Optional[str]:
        """Return the model's text, or None when offline / on any error (callers have fallbacks)."""
        if not self.enabled:
            return None
        try:
            if self.provider == "groq":
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "temperature": 0.2, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={"model": self.model, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except Exception as exc:  # network, quota, bad key... never crash the pipeline
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
