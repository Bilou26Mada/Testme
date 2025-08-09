"""Utility functions and constants for CLI LLM provider selection.

This module defines available LLM providers and helper functions to
select shallow or deep thinking agents for a given provider.
"""

from __future__ import annotations

from typing import Dict, List

# A list of available base LLM providers.
BASE_URLS: List[str] = [
    "OpenAI",
    "Anthropic",
    "DeepSeek",
]

# Mapping of provider key to available shallow thinking agent models.
SHALLOW_AGENT_OPTIONS: Dict[str, List[str]] = {
    "openai": ["gpt-4o-mini"],
    "anthropic": ["claude-3-haiku"],
    "deepseek": ["deepseek-chat"],
}

# Mapping of provider key to available deep thinking agent models.
DEEP_AGENT_OPTIONS: Dict[str, List[str]] = {
    "openai": ["gpt-4o"],
    "anthropic": ["claude-3-opus"],
    "deepseek": ["deepseek-reasoner"],
}


def select_llm_provider(provider_key: str) -> str:
    """Return the provider name for a given key.

    Parameters
    ----------
    provider_key:
        Key identifying the provider (case-insensitive).

    Returns
    -------
    str
        The human-readable provider name.

    Raises
    ------
    KeyError
        If the provider key is unknown.
    """

    providers = {name.lower(): name for name in BASE_URLS}
    key = provider_key.lower()
    if key not in providers:
        raise KeyError(f"Unknown LLM provider: {provider_key}")
    return providers[key]


def select_shallow_thinking_agent(provider_key: str) -> str:
    """Return the shallow thinking agent for the specified provider.

    Parameters
    ----------
    provider_key:
        Key identifying the provider (case-insensitive).

    Returns
    -------
    str
        The name of the shallow thinking agent model.

    Raises
    ------
    KeyError
        If the provider key is unknown.
    """

    key = provider_key.lower()
    try:
        return SHALLOW_AGENT_OPTIONS[key][0]
    except KeyError as exc:
        raise KeyError(f"Unknown provider for shallow agent: {provider_key}") from exc


def select_deep_thinking_agent(provider_key: str) -> str:
    """Return the deep thinking agent for the specified provider.

    Parameters
    ----------
    provider_key:
        Key identifying the provider (case-insensitive).

    Returns
    -------
    str
        The name of the deep thinking agent model.

    Raises
    ------
    KeyError
        If the provider key is unknown.
    """

    key = provider_key.lower()
    try:
        return DEEP_AGENT_OPTIONS[key][0]
    except KeyError as exc:
        raise KeyError(f"Unknown provider for deep agent: {provider_key}") from exc
