"""
Model provider resolution for Pydantic AI.

This module handles the conversion of ChainLite model names to Pydantic AI model strings.
"""

from typing import Optional
from dataclasses import dataclass


# Mapping from ChainLite provider prefixes to Pydantic AI model prefixes
PROVIDER_MAPPING = {
    "openai:": "openai:",
    "google-vertexai:": "google-vertex:",
    "google-gemini:": "google-gla:",
    "anthropic:": "anthropic:",
    "mistralai:": "mistral:",
    "ollama:": "ollama:",  # Uses OpenAI-compatible endpoint
}


def resolve_model_string(llm_model_name: str) -> str:
    """
    Converts a ChainLite model name to a Pydantic AI model string.

    Args:
        llm_model_name: The model name with provider prefix (e.g., "openai:gpt-4")

    Returns:
        The Pydantic AI compatible model string.

    Raises:
        ValueError: If the provider prefix is not recognized.

    Examples:
        >>> resolve_model_string("openai:gpt-4")
        'openai:gpt-4'
        >>> resolve_model_string("google-gemini:gemini-pro")
        'google-gla:gemini-pro'
        >>> resolve_model_string("anthropic:claude-3-sonnet")
        'anthropic:claude-3-sonnet'
    """
    for chainlite_prefix, pydantic_prefix in PROVIDER_MAPPING.items():
        if llm_model_name.startswith(chainlite_prefix):
            model_name = llm_model_name[len(chainlite_prefix) :]
            return f"{pydantic_prefix}{model_name}"

    raise ValueError(
        f"Unknown model provider: {llm_model_name}. "
        f"Supported prefixes: {list(PROVIDER_MAPPING.keys())}"
    )
