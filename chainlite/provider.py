"""
Model provider resolution for Pydantic AI.

This module handles the conversion of ChainLite model names to Pydantic AI model strings.
It uses a default mapping but allows user configuration to extend or override it.
"""

from typing import Optional, Dict


# Default mapping from ChainLite provider prefixes to Pydantic AI model prefixes
DEFAULT_MAPPING = {
    "openai:": "openai:",
    "google-vertexai:": "google-vertex:",
    "google-gemini:": "google-gla:",
    "anthropic:": "anthropic:",
    "mistralai:": "mistral:",
    "ollama:": "ollama:",  # Uses OpenAI-compatible endpoint
}


def resolve_model_string(
    llm_model_name: str, custom_mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Resolves the model string for Pydantic AI.

    It merges the `DEFAULT_MAPPING` with any provided `custom_mapping`.
    If a `custom_mapping` is provided, it takes precedence over the defaults.

    Args:
        llm_model_name: The model name string (e.g., "my-alias:gpt-4" or "openai:gpt-4").
        custom_mapping: Optional dictionary mapping user prefixes to Pydantic AI prefixes.
                        Example: {"gemini:": "google-gla:gemini-1.5-"}

    Returns:
        The resolved model string compatible with Pydantic AI.

    Examples:
        # Default behavior
        >>> resolve_model_string("google-gemini:gemini-pro")
        'google-gla:gemini-pro'

        # Custom extension (adding an alias)
        >>> mapping = {"my-gpt:": "openai:gpt-4-"}
        >>> resolve_model_string("my-gpt:turbo", mapping)
        'openai:gpt-4-turbo'

        # Custom override (changing default behavior)
        >>> mapping = {"openai:": "azure-openai:"}
        >>> resolve_model_string("openai:gpt-4", mapping)
        'azure-openai:gpt-4'
    """
    # 1. Create a base mapping from defaults (Copy to avoid mutating global state)
    final_mapping = DEFAULT_MAPPING.copy()

    # 2. Merge user-defined mapping if provided
    # This allows users to add new aliases or override existing defaults
    if custom_mapping:
        final_mapping.update(custom_mapping)

    # 3. Sort keys by length (descending) to ensure longest prefix match
    # This prevents partial matches (e.g., properly distinguishing between "gpt:" and "gpt-4:")
    sorted_providers = sorted(
        final_mapping.items(), key=lambda x: len(x[0]), reverse=True
    )

    for user_prefix, target_prefix in sorted_providers:
        if llm_model_name.startswith(user_prefix):
            # Calculate the suffix (the part after the prefix)
            model_suffix = llm_model_name[len(user_prefix) :]
            # Return the new constructed string
            return f"{target_prefix}{model_suffix}"

    # 4. Fallback: If no prefix matches, return as is (Pass-through mode)
    return llm_model_name
