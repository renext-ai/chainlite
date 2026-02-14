"""Agent setup and initialization helpers for ChainLite."""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from .adapters.pydantic_ai import get_agent_tools
from .compaction import InRunCompactionConfig


def build_compaction_components(
    config: Any,
) -> tuple[Any, int, Optional[InRunCompactionConfig], Any]:
    """Build post-run and in-run compaction components from config."""
    truncator = None
    post_run_compaction_start_run = 1
    in_run_compaction_config: Optional[InRunCompactionConfig] = None
    in_run_compactor = None

    if not config.history_truncator_config:
        return truncator, post_run_compaction_start_run, in_run_compaction_config, in_run_compactor

    from .truncators import SimpleTruncator, AutoSummarizer, ChainLiteSummarizer

    t_config = config.history_truncator_config
    post_cfg = t_config.post_run_compaction
    threshold = 5000

    if post_cfg and post_cfg.mode:
        mode = post_cfg.mode
        threshold = post_cfg.truncation_threshold
        post_run_compaction_start_run = post_cfg.start_run

        if mode == "simple":
            truncator = SimpleTruncator(threshold=threshold)
        elif mode == "auto":
            truncator = AutoSummarizer(
                threshold=threshold,
                model_name=config.llm_model_name,
            )
        elif mode == "custom":
            truncator = ChainLiteSummarizer(
                config_or_path=(
                    post_cfg.summarizer_config_path
                    or post_cfg.summarizer_config_dict
                ),
                threshold=threshold,
            )

    in_run_cfg = t_config.in_run_compaction
    if in_run_cfg and in_run_cfg.mode:
        in_run_mode = in_run_cfg.mode
        in_run_threshold = in_run_cfg.truncation_threshold or threshold
        in_run_compaction_config = {
            "start_iter": in_run_cfg.start_iter,
            "start_run": in_run_cfg.start_run,
            "max_concurrency": in_run_cfg.max_concurrency,
        }

        if in_run_mode == "simple":
            in_run_compactor = SimpleTruncator(threshold=in_run_threshold)
        elif in_run_mode == "auto":
            in_run_compactor = AutoSummarizer(
                threshold=in_run_threshold,
                model_name=config.llm_model_name,
            )
        elif in_run_mode == "custom":
            in_run_compactor = ChainLiteSummarizer(
                config_or_path=(
                    in_run_cfg.summarizer_config_path
                    or in_run_cfg.summarizer_config_dict
                ),
                threshold=in_run_threshold,
            )

        logger.info(
            f"In-run compaction enabled: mode={in_run_mode}, threshold={in_run_threshold}, "
            f"start_iter={in_run_compaction_config['start_iter']}, "
            f"start_run={in_run_compaction_config['start_run']}, "
            f"max_concurrency={in_run_compaction_config['max_concurrency']}"
        )

    return (
        truncator,
        post_run_compaction_start_run,
        in_run_compaction_config,
        in_run_compactor,
    )


def build_model_settings(config: Any) -> Optional[ModelSettings]:
    """Build pydantic-ai model settings from ChainLite config."""
    settings_dict: dict[str, Any] = {}

    if config.model_settings:
        settings_dict.update(config.model_settings)

    if config.temperature is not None:
        settings_dict["temperature"] = config.temperature

    if not settings_dict:
        return None

    return ModelSettings(**settings_dict)


def create_agent_instance(
    *,
    model_string: str,
    instructions: Optional[str],
    output_model: Any,
    retries: int,
    tools: Optional[list[Any]] = None,
) -> Agent:
    """Create a configured pydantic-ai agent instance."""
    kwargs: dict[str, Any] = {
        "instructions": instructions,
        "retries": retries,
    }

    if output_model:
        kwargs["output_type"] = output_model

    if tools is not None:
        kwargs["tools"] = tools

    return Agent(model_string, **kwargs)


def collect_agent_tools(agent: Optional[Agent]) -> list[Any]:
    """Collect registered tools from a pydantic-ai agent (version-compatible)."""
    return get_agent_tools(agent)
