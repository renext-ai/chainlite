"""Compatibility helpers for pydantic-ai public/private API differences.

This adapter intentionally centralizes access to unstable/private pydantic-ai
internals so version drift can be fixed in one place.
"""

from __future__ import annotations

import json
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


def get_agent_tool_schemas(agent: Optional[Any]) -> list[dict[str, Any]]:
    """Return normalized tool schema dictionaries for auditing/export."""
    schemas: list[dict[str, Any]] = []
    for tool in get_agent_tools(agent):
        name = getattr(tool, "name", "unknown")
        description = getattr(tool, "description", None)

        schema_data: dict[str, Any] = {
            "name": name,
            "description": description,
        }

        function_schema = getattr(tool, "function_schema", None)
        if function_schema is not None:
            schema_data["parameters"] = getattr(function_schema, "json_schema", {})

        schemas.append(schema_data)
    return schemas


def get_agent_instructions(agent: Optional[Any]) -> list[str]:
    """Return normalized string instructions from an agent."""
    if not agent:
        return []

    instructions = getattr(agent, "_instructions", None)
    if not instructions:
        return []

    return [str(item) for item in instructions]


def build_full_system_prompt(agent: Optional[Any]) -> str:
    """Build a system prompt string with instructions and tool schemas."""
    if not agent:
        return ""

    lines: list[str] = []

    instructions = get_agent_instructions(agent)
    if instructions:
        lines.append("## Instructions\n")
        for inst in instructions:
            lines.append(f"{inst}\n")

    tool_schemas = get_agent_tool_schemas(agent)
    if tool_schemas:
        if lines:
            lines.append("\n")
        lines.append("## Tools\n")
        for schema_data in tool_schemas:
            name = schema_data.get("name", "unknown")
            description = schema_data.get("description")
            lines.append(f"\n### Tool: {name}")
            if description:
                lines.append(f"**Description**: {description}")

            lines.append("```json")
            lines.append(json.dumps(schema_data, indent=2, ensure_ascii=False))
            lines.append("```")

    return "\n".join(lines)


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


def is_model_request_node(node: Any) -> bool:
    """Best-effort detection for pydantic-ai model-request graph nodes."""
    if node is None:
        return False

    try:
        from pydantic_ai.agent import ModelRequestNode  # type: ignore

        return isinstance(node, ModelRequestNode)
    except Exception:
        pass

    node_name = type(node).__name__
    return node_name == "ModelRequestNode"
