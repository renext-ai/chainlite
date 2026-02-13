"""Compatibility helpers for pydantic-ai public/private API differences."""

from __future__ import annotations

from typing import Any, Optional


def get_agent_tools(agent: Optional[Any]) -> list[Any]:
    """Return registered tools from an agent across known pydantic-ai layouts."""
    if not agent:
        return []

    # Preferred path in recent versions.
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is not None:
        tools = getattr(toolset, "tools", None)
        if isinstance(tools, dict):
            return list(tools.values())

    # Legacy path in older versions.
    legacy_tools = getattr(agent, "_function_tools", None)
    if isinstance(legacy_tools, dict):
        return list(legacy_tools.values())

    return []


def get_agent_instructions(agent: Optional[Any]) -> list[str]:
    """Return normalized string instructions from an agent."""
    if not agent:
        return []

    instructions = getattr(agent, "_instructions", None)
    if not instructions:
        return []

    return [str(item) for item in instructions]


def is_call_tools_node(node: Any) -> bool:
    """Best-effort detection for pydantic-ai call-tools graph nodes."""
    if node is None:
        return False

    # Fast path when class is importable.
    try:
        from pydantic_ai.agent import CallToolsNode  # type: ignore

        return isinstance(node, CallToolsNode)
    except Exception:
        pass

    # Fallback path for version drift where class import/membership changes.
    node_name = type(node).__name__
    return node_name == "CallToolsNode"

