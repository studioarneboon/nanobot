"""Provider factory — creates LLM provider instances from config.

Extracted from cli/commands.py so that AgentLoop can instantiate providers
at runtime (e.g. for model fallback switching without a full restart).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider


def make_provider(
    config: "Config",
    model: str | None = None,
    provider_name: str | None = None,
) -> "LLMProvider":
    """Create the appropriate LLM provider for the given model.

    Args:
        config: Root nanobot config object.
        model: Model name override. Defaults to config.agents.defaults.model.
        provider_name: Force a specific provider. Defaults to auto-detection.

    Returns:
        An LLMProvider instance ready for chat() calls.

    Raises:
        SystemExit: If no API key is configured (same behaviour as CLI).
    """
    from nanobot.providers.custom_provider import CustomProvider
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.registry import find_by_name

    effective_model = model or config.agents.defaults.model

    # When a provider_name is forced, temporarily override the config's provider
    # setting so _match_provider() resolves correctly.
    if provider_name and provider_name != "auto":
        # Build a minimal override: store original, swap, call, restore.
        original_provider = config.agents.defaults.provider
        config.agents.defaults.provider = provider_name
        p = config.get_provider(effective_model)
        api_base = config.get_api_base(effective_model)
        resolved_provider_name = config.get_provider_name(effective_model)
        config.agents.defaults.provider = original_provider
    else:
        p = config.get_provider(effective_model)
        api_base = config.get_api_base(effective_model)
        resolved_provider_name = config.get_provider_name(effective_model)

    # OpenAI Codex (OAuth)
    if resolved_provider_name == "openai_codex" or effective_model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=effective_model)

    # Custom / opencode / lmstudio / ollama: direct OpenAI-compatible endpoints
    # These bypass LiteLLM and talk to the endpoint directly.
    direct_providers = {"custom", "opencode", "lmstudio", "ollama"}
    if resolved_provider_name in direct_providers:
        return CustomProvider(
            api_key=p.api_key if (p and p.api_key) else "no-key",
            api_base=api_base or "http://localhost:8000/v1",
            default_model=effective_model,
        )

    spec = find_by_name(resolved_provider_name) if resolved_provider_name else None
    if not effective_model.startswith("bedrock/") and not (p and p.api_key) and not (spec and spec.is_oauth):
        from loguru import logger
        logger.error(
            "No API key configured for model '{}' (provider: {}). "
            "Set one in ~/.nanobot/config.json under providers section.",
            effective_model,
            resolved_provider_name,
        )
        raise ValueError(
            f"No API key configured for model '{effective_model}' (provider: {resolved_provider_name})"
        )

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=api_base,
        default_model=effective_model,
        extra_headers=p.extra_headers if p else None,
        provider_name=resolved_provider_name,
    )
