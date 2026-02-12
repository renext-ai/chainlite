from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent that will be registered as a tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """The tool name exposed to the parent agent (e.g. 'translate')."""

    config: str
    """Path to the sub-agent's YAML config file.
    Resolved relative to the parent YAML when using load_config_from_yaml().
    """

    description: str
    """Description shown to the LLM explaining when/how to use this tool."""


class ChainLiteConfig(BaseModel):
    """
    Configuration model for ChainLite.
    """

    model_config = ConfigDict(extra="forbid")

    config_name: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    llm_model_name: str
    model_settings: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    output_parser: Optional[List[Dict[str, Any]]] = None
    use_history: Optional[bool] = False
    session_id: str = "unused"
    redis_url: Optional[str] = None
    max_retries: Optional[int] = 1
    # History Truncation and Summarization
    # mode: "simple" | "auto" | "custom"
    history_truncator_config: Optional[Dict[str, Any]] = None
    # Sub-agents: declared here and auto-registered as tools on the parent agent
    sub_agents: Optional[List[SubAgentConfig]] = None
