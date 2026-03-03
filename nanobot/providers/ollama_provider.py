"""Ollama provider — uses native Ollama API (/api/chat) for maximum speed.

The OpenAI-compatible /v1/chat/completions endpoint on ollama is ~30x slower
than the native /api/chat endpoint for CPU inference. This provider uses the
native endpoint directly via httpx.
"""

from __future__ import annotations

from typing import Any

import httpx

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OllamaProvider(LLMProvider):
    """Direct Ollama native API provider — fast even on CPU."""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:11434",
        default_model: str = "qwen2.5:0.5b",
    ):
        # Strip trailing /v1 if accidentally passed (from config)
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        super().__init__(api_key="no-key", api_base=base)
        self.default_model = default_model
        self._client = httpx.AsyncClient(base_url=base, timeout=90.0)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        effective_model = model or self.default_model

        # Convert messages: ollama native uses same role/content format as OpenAI
        clean_messages = self._sanitize_empty_content(messages)
        ollama_messages = [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in clean_messages
            if m.get("role") in ("system", "user", "assistant")
        ]

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            done_reason = data.get("done_reason", "stop")
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
            return LLMResponse(
                content=content or None,
                tool_calls=[],
                finish_reason=done_reason or "stop",
                usage=usage,
            )
        except httpx.TimeoutException as e:
            return LLMResponse(content=f"Error: ollama timeout: {e}", finish_reason="rate_limit")
        except Exception as e:
            error_str = str(e).lower()
            rate_limit_signals = ("rate limit", "429", "overloaded", "capacity", "unavailable", "503", "502")
            if any(sig in error_str for sig in rate_limit_signals):
                return LLMResponse(content=f"Error: {e}", finish_reason="rate_limit")
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model
