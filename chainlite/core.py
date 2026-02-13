"""
Core ChainLite module using Pydantic AI.

This module provides a lightweight wrapper around Pydantic AI's Agent for building
LLM-powered applications with support for streaming, structured output, and
conversation history.
"""

from typing import Optional, Generator, AsyncGenerator, Dict, List, Any, Union
import asyncio
import traceback
import json
from loguru import logger
import yaml
from pydantic import BaseModel
import jinja2

from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.messages import ModelMessage

from .config import ChainLiteConfig
from .provider import resolve_model_string
from .history import HistoryManager
from .streaming import StreamProcessor
from ._factories import (
    build_compaction_components,
    build_model_settings,
    collect_agent_tools,
    create_agent_instance,
)
from .utils.media import build_prompt
from .utils.output_model import (
    create_dynamic_pydantic_model,
    merge_dictionaries,
)
from .utils.prompts import parse_input_variables_from_prompt
from .adapters.pydantic_ai import (
    get_agent_instructions,
    get_agent_tools,
    is_call_tools_node,
)


class ChainLite:
    """Main class for a ChainLite instance.

    This class handles the ChainLite configuration and provides methods for setting up
    the chat flow and executing it using Pydantic AI.

    Attributes:
        config: An instance of ChainLiteConfig that holds all the necessary configuration.
        agent: The Pydantic AI Agent instance.
        history_manager: Optional manager for conversation history.
    """

    def __init__(self, config: ChainLiteConfig) -> None:
        """Initializes a new ChainLite instance.

        Args:
            config: A ChainLiteConfig instance containing all the necessary configuration.
        """
        self.config = config
        self.agent: Optional[Agent] = None
        self.output_model: Optional[type[BaseModel]] = None
        self.history_manager: Optional[HistoryManager] = None
        self.exception_retry = 0
        self._truncator = None
        self._post_run_compaction_start_run = 1
        self._in_run_compaction_config = None
        self._in_run_compactor = None
        # Backward-compatible aliases for older integrations.
        self._lazy_config = None
        self._lazy_truncator = None
        self._run_count = 0

        self.setup_chain()

    def setup_chain(self) -> None:
        """Sets up the Pydantic AI Agent.

        This method initializes the agent with the provided configuration.
        """
        # Resolve model string for Pydantic AI
        model_string = resolve_model_string(self.config.llm_model_name)

        logger.info(
            f"Using {self.config.llm_model_name} (resolved: {model_string})"
            f"{' (config: ' + self.config.config_name + ')' if self.config.config_name else ''}"
        )

        # Build output model if output_parser is configured
        if self.config.output_parser:
            self.output_model = create_dynamic_pydantic_model(
                merge_dictionaries(self.config.output_parser)
            )

        (
            truncator,
            self._post_run_compaction_start_run,
            self._in_run_compaction_config,
            self._in_run_compactor,
        ) = build_compaction_components(self.config)
        self._truncator = truncator
        # Keep legacy internal names synchronized for compatibility.
        self._lazy_config = self._in_run_compaction_config
        self._lazy_truncator = self._in_run_compactor

        if self.config.use_history:
            logger.info(
                f"Chain Builder using history with session_id: {self.config.session_id}"
                f"{' (config: ' + self.config.config_name + ')' if self.config.config_name else ''}"
            )
            self.history_manager = HistoryManager(
                session_id=self.config.session_id,
                redis_url=self.config.redis_url,
                truncator=truncator,
            )
            # Set initial system prompt if agent is already set up (unlikely here but for safety)
            if self.agent:
                self.history_manager.system_prompt = self._get_full_system_prompt()

        # Build system instructions
        instructions = self._build_instructions()

        self._model_settings = build_model_settings(self.config)
        self.agent = create_agent_instance(
            model_string=model_string,
            instructions=instructions,
            output_model=self.output_model,
            retries=self.config.max_retries or 3,
        )

    def _create_agent(self, instructions: Optional[str]) -> Agent:
        """Create an agent instance with the given instructions."""
        model_string = resolve_model_string(self.config.llm_model_name)
        tools = collect_agent_tools(self.agent)
        return create_agent_instance(
            model_string=model_string,
            instructions=instructions,
            output_model=self.output_model,
            retries=self.config.max_retries or 3,
            tools=tools,
        )

    def _build_instructions(self, context: Dict[str, Any] = None) -> Optional[str]:
        """Build system instructions from config, optionally rendering with context."""
        parts = []

        if self.config.system_prompt:
            system_prompt = self.config.system_prompt
            if context:
                # Check if system prompt has variables before attempting render
                # This avoids unnecessary processing if it's static
                if parse_input_variables_from_prompt(system_prompt):
                    try:
                        template = jinja2.Template(system_prompt)
                        system_prompt = template.render(**context)
                    except Exception as e:
                        logger.warning(f"Failed to render system_prompt: {e}")

            parts.append(system_prompt)

        if parts:
            return "\n\n".join(parts)
        return None

    async def _build_prompt(
        self, input_data: dict
    ) -> Union[str, list[Union[str, BinaryContent, ImageUrl]]]:
        """Build the user prompt from template and input data."""
        prompt_template = self.config.prompt or "{{ input }}"

        try:
            return await build_prompt(prompt_template, input_data)
        except Exception as e:
            logger.warning(f"Error in prompt formatting: {e}")
            raise e

    async def _prepare_run(self, input_data: dict) -> tuple[
        Union[str, list[Union[str, BinaryContent, ImageUrl]]],
        Optional[List[ModelMessage]],
    ]:
        """Prepare for agent run: build prompt and get history."""
        if self.agent is None:
            raise ValueError(
                "Agent is not set up. Please call `setup_chain` method before running."
            )

        # Ensure HistoryManager has the full system prompt (including dynamically added tools)
        if self.history_manager and not self.history_manager.system_prompt:
            self.history_manager.system_prompt = self._get_full_system_prompt()

        prompt = await self._build_prompt(input_data)
        if self.history_manager and self.history_manager.messages:
            message_history = self.history_manager.messages
        else:
            message_history = None
        return prompt, message_history

    def _process_run_result(
        self, result: Any, context: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Process agent run result: update history and dump output."""
        if self.history_manager:
            self.history_manager.add_messages(
                result.new_messages(),
                context=context,
                apply_truncation=self._should_apply_post_run_compaction(),
            )

        output = result.output
        if isinstance(output, BaseModel):
            return output.model_dump()
        return output

    async def _aprocess_run_result(
        self, result: Any, context: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Process agent run result asynchronously."""
        if self.history_manager:
            await self.history_manager.add_messages_async(
                result.new_messages(),
                context=context,
                apply_truncation=self._should_apply_post_run_compaction(),
            )

        output = result.output
        if isinstance(output, BaseModel):
            return output.model_dump()
        return output

    def _handle_run_error(
        self,
        e: Exception,
        context: str,
        prompt: Union[str, list[Union[str, BinaryContent, ImageUrl]]],
        input_data: dict,
    ) -> None:
        """Log error and handle retry state."""
        logger.warning(f"Error in {context}:\n{traceback.format_exc()}")
        logger.warning(f"Prompt: {prompt}")
        logger.warning(f"Input: {input_data}")
        self.exception_retry += 1

    async def _get_agent_for_run(self, input_data: dict) -> Agent:
        """Get the appropriate agent for the run, creating a dynamic one if needed."""
        if self.agent is None:
            raise ValueError(
                "Agent is not set up. Please call `setup_chain` method before running."
            )

        # Check if system prompt needs dynamic rendering
        if self.config.system_prompt and parse_input_variables_from_prompt(
            self.config.system_prompt
        ):
            instructions = self._build_instructions(input_data)
            return self._create_agent(instructions)

        return self.agent

    def _should_apply_post_run_compaction(self) -> bool:
        """Check if post-run compaction should be applied for this run."""
        return self._run_count >= self._post_run_compaction_start_run

    def _should_use_in_run_compaction(self) -> bool:
        """Check if in-run compaction should be used for the current run."""
        return (
            self.history_manager is not None
            and self._in_run_compaction_config is not None
            and self._in_run_compactor is not None
            and self._run_count >= self._in_run_compaction_config["start_run"]
        )

    async def _arun_with_in_run_compaction(
        self, agent, prompt, message_history, model_settings, deps, context
    ):
        """Run agent with in-run compaction of previous tool results."""
        from pydantic_graph import End

        tool_iter_count = 0
        in_run_task: Optional[asyncio.Task] = None
        in_run_refresh_requested = False

        def _schedule_in_run_if_needed(messages: list) -> None:
            nonlocal in_run_task, in_run_refresh_requested
            if in_run_task is None or in_run_task.done():
                in_run_task = asyncio.create_task(
                    self.history_manager.apply_in_run_compaction_to_previous_tool_results(
                        messages,
                        compactor=self._in_run_compactor,
                        context=context,
                        max_concurrency=self._in_run_compaction_config.get(
                            "max_concurrency", 4
                        ),
                    )
                )
                in_run_refresh_requested = False
            else:
                # Coalesce repeated triggers while one in-run task is in flight.
                in_run_refresh_requested = True

        async with agent.iter(
            prompt,
            message_history=message_history,
            model_settings=model_settings,
            deps=deps,
        ) as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                if is_call_tools_node(node):
                    tool_iter_count += 1
                    if (
                        tool_iter_count
                        >= self._in_run_compaction_config["start_iter"]
                    ):
                        _schedule_in_run_if_needed(agent_run.all_messages())
                node = await agent_run.next(node)

                # If a pending refresh was requested while the previous task
                # was running, immediately schedule another pass when it completes.
                if (
                    in_run_task is not None
                    and in_run_task.done()
                    and in_run_refresh_requested
                ):
                    _schedule_in_run_if_needed(agent_run.all_messages())

            # Final synchronization: ensure outstanding in-run work is applied before
            # returning result so history persists the latest summarized state.
            if in_run_task is not None:
                await in_run_task
            if in_run_refresh_requested:
                await self.history_manager.apply_in_run_compaction_to_previous_tool_results(
                    agent_run.all_messages(),
                    compactor=self._in_run_compactor,
                    context=context,
                    max_concurrency=self._in_run_compaction_config.get(
                        "max_concurrency", 4
                    ),
                )

        return agent_run.result

    async def _arun_core(
        self,
        input_data: dict,
        deps: Any = None,
        *,
        error_context: str,
        retry_sleep_seconds: float,
        use_async_result_processing: bool,
    ) -> Union[str, Dict[str, Any]]:
        """Shared async core for run/arun execution flow."""
        prompt, message_history = await self._prepare_run(input_data)

        # Get agent (potentially dynamic)
        agent = await self._get_agent_for_run(input_data)
        self._run_count += 1
        input_str = input_data.get("input") or str(prompt)

        max_retries = self.config.max_retries or 3
        current_try = 0

        while True:
            try:
                if self._should_use_in_run_compaction():
                    result = await self._arun_with_in_run_compaction(
                        agent,
                        prompt,
                        message_history,
                        self._model_settings,
                        deps,
                        input_str,
                    )
                else:
                    result = await agent.run(
                        prompt,
                        message_history=message_history,
                        model_settings=self._model_settings,
                        deps=deps,
                    )

                if use_async_result_processing:
                    return await self._aprocess_run_result(result, context=input_str)
                return self._process_run_result(result, context=input_str)

            except Exception as e:
                current_try += 1
                self._handle_run_error(e, error_context, prompt, input_data)

                if current_try >= max_retries:
                    logger.error("Max retries reached")
                    raise e

                logger.info(f"Retrying {current_try}/{max_retries}")
                if retry_sleep_seconds > 0:
                    await asyncio.sleep(retry_sleep_seconds)

    def run(self, input_data: dict, deps: Any = None) -> Union[str, Dict[str, Any]]:
        """
        Executes the synchronous chat flow.

        Args:
            input_data: A dictionary containing the input data for the chat flow.
            deps: Optional dependencies to pass to the agent.

        Returns:
            The response from the language model, either as string or structured data.
        """
        return asyncio.run(
            self._arun_core(
                input_data,
                deps,
                error_context="run",
                retry_sleep_seconds=0,
                use_async_result_processing=False,
            )
        )

    async def arun(
        self, input_data: dict, deps: Any = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Executes the asynchronous chat flow.

        Args:
            input_data: A dictionary containing the input data for the chat flow.
            deps: Optional dependencies to pass to the agent.

        Returns:
            The response from the language model, either as string or structured data.
        """
        return await self._arun_core(
            input_data,
            deps,
            error_context="arun",
            retry_sleep_seconds=1,
            use_async_result_processing=True,
        )

    def stream(self, input_data: dict, deps: Any = None) -> Generator[str, None, None]:
        """
        Synchronously processes input data through the agent, yielding text chunks.

        Args:
            input_data: A dictionary of input data.
            deps: Optional dependencies to pass to the agent.

        Yields:
            Text chunks from the streaming response.
        """
        prompt, message_history = asyncio.run(self._prepare_run(input_data))

        # Get agent (potentially dynamic)
        agent = asyncio.run(self._get_agent_for_run(input_data))

        result = agent.run_stream_sync(
            prompt,
            message_history=message_history,
            model_settings=self._model_settings,
            deps=deps,
        )

        processor = StreamProcessor()

        if hasattr(result, "stream_responses"):
            for message in result.stream_responses():
                if isinstance(message, tuple):
                    message = message[0]
                for chunk in processor.process_message(message):
                    yield chunk
        elif hasattr(result, "stream_structured"):
            for message in result.stream_structured():
                if isinstance(message, tuple):
                    message = message[0]
                for chunk in processor.process_message(message):
                    yield chunk
        else:
            for chunk in result.stream_text(delta=True):
                yield chunk

        # Close any open tags
        for chunk in processor.close():
            yield chunk

        # Update history after stream completes
        if self.history_manager:
            self.history_manager.add_messages(
                result.new_messages(),
                apply_truncation=self._should_apply_post_run_compaction(),
            )

    async def astream(
        self, input_data: dict, deps: Any = None
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously processes input data through the agent, yielding text chunks.
        Uses _StreamProcessor to handle part transitions and inject headers/tags.
        """
        prompt, message_history = await self._prepare_run(input_data)
        agent = await self._get_agent_for_run(input_data)

        async with agent.run_stream(
            prompt,
            message_history=message_history,
            model_settings=self._model_settings,
            deps=deps,
        ) as result:

            processor = StreamProcessor()

            if hasattr(result, "stream_responses"):
                async for message in result.stream_responses():
                    if isinstance(message, tuple):
                        message = message[0]

                    for chunk in processor.process_message(message):
                        yield chunk
            elif hasattr(result, "stream_structured"):
                async for message in result.stream_structured():
                    if isinstance(message, tuple):
                        message = message[0]

                    for chunk in processor.process_message(message):
                        yield chunk
            else:
                async for chunk in result.stream_text(delta=True):
                    yield chunk

            # Close any open tags
            for chunk in processor.close():
                yield chunk

            # Update history after stream completes
            # Update history after stream completes
            if self.history_manager:
                input_str = input_data.get("input") or str(prompt)
                await self.history_manager.add_messages_async(
                    result.new_messages(),
                    context=input_str,
                    apply_truncation=self._should_apply_post_run_compaction(),
                )

    def _get_full_system_prompt(self) -> str:
        """Extract instructions and tool definitions from the agent."""
        if not self.agent:
            return ""

        lines = []
        # 1. Instructions
        instructions = get_agent_instructions(self.agent)
        if instructions:
            lines.append("## Instructions\n")
            for inst in instructions:
                lines.append(f"{inst}\n")

        # 2. Tools
        tools = get_agent_tools(self.agent)
        if tools:
            if lines:
                lines.append("\n")
            lines.append("## Tools\n")
            for tool in tools:
                name = getattr(tool, "name", "unknown")
                lines.append(f"\n### Tool: {name}")
                description = getattr(tool, "description", None)
                if description:
                    lines.append(f"**Description**: {description}")

                # Build a complete tool schema for audit
                schema_data = {
                    "name": name,
                    "description": description,
                }
                if hasattr(tool, "function_schema"):
                    # Use json_schema from pydantic-ai's FunctionSchema
                    schema_data["parameters"] = getattr(
                        tool.function_schema, "json_schema", {}
                    )

                lines.append("```json")
                lines.append(json.dumps(schema_data, indent=2, ensure_ascii=False))
                lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def load_config_from_yaml(
        yaml_file_path: str, custom_configs: dict = None
    ) -> "ChainLite":
        """
        Load and return a ChainLite instance from a YAML file.

        Args:
            yaml_file_path: The file path of the YAML configuration file.
            custom_configs: A dictionary of configurations that overrides the
                           settings from the YAML file.

        Returns:
            A ChainLite instance initialized with the settings from the YAML file.
        """
        with open(yaml_file_path, "r", encoding="utf-8") as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

        if custom_configs is not None:
            yaml_data.update(custom_configs)

        if yaml_data.get("config_name") is None:
            yaml_data["config_name"] = str(yaml_file_path)

        return ChainLite(ChainLiteConfig(**yaml_data))
