"""
Core ChainLite module using Pydantic AI.

This module provides a lightweight wrapper around Pydantic AI's Agent for building
LLM-powered applications with support for streaming, structured output, and
conversation history.
"""

from typing import (
    Optional,
    Generator,
    AsyncGenerator,
    Dict,
    List,
    Any,
    Union,
)
import asyncio
import traceback
import json
from loguru import logger
import yaml
from pydantic import BaseModel
import jinja2

from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from .config import ChainLiteConfig
from .provider import resolve_model_string
from .history import HistoryManager
from ._in_run_compaction import InRunCompactionTaskManager
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
    get_agent_tool_schemas,
    is_call_tools_node,
    is_model_request_node,
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

    def _count_tool_result_blocks(self, messages: List[ModelMessage]) -> int:
        """Count ModelRequest blocks that contain ToolReturnPart."""
        count = 0
        for msg in messages:
            if isinstance(msg, ModelRequest) and any(
                isinstance(p, ToolReturnPart) for p in msg.parts
            ):
                count += 1
        return count

    def _should_include_latest_for_in_run(self) -> bool:
        """For start_iter=1, allow compacting the latest tool block."""
        return (
            self._in_run_compaction_config is not None
            and int(self._in_run_compaction_config.get("start_iter", 2)) <= 1
        )

    async def _apply_stream_in_run_compaction_if_needed(
        self,
        messages: List[ModelMessage],
        context: Optional[str],
    ) -> None:
        """Apply in-run compaction for streaming runs before persisting history."""
        if not self._should_use_in_run_compaction():
            return

        tool_blocks = self._count_tool_result_blocks(messages)
        if tool_blocks < self._in_run_compaction_config["start_iter"]:
            return

        await self._run_in_run_compaction(messages, context)

    async def _run_in_run_compaction(
        self,
        messages: List[ModelMessage],
        context: Optional[str],
    ) -> None:
        """Run in-run compaction over provided messages."""
        await self.history_manager.apply_in_run_compaction_to_previous_tool_results(
            messages,
            compactor=self._in_run_compactor,
            context=context,
            max_concurrency=self._in_run_compaction_config.get("max_concurrency", 4),
            include_latest_tool_block=self._should_include_latest_for_in_run(),
        )

    def _build_in_run_compaction_task_manager(
        self, context: Optional[str]
    ) -> InRunCompactionTaskManager:
        """Create task manager bound to the current run context."""

        async def _apply_in_run_compaction(messages: List[ModelMessage]) -> None:
            await self._run_in_run_compaction(messages, context)

        return InRunCompactionTaskManager(
            start_iter=self._in_run_compaction_config["start_iter"],
            apply_compaction=_apply_in_run_compaction,
        )

    async def _arun_with_in_run_compaction(
        self, agent, prompt, message_history, model_settings, deps, context
    ):
        """Run agent with in-run compaction of previous tool results."""
        from pydantic_graph import End

        task_manager = self._build_in_run_compaction_task_manager(context)

        async with agent.iter(
            prompt,
            message_history=message_history,
            model_settings=model_settings,
            deps=deps,
        ) as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                was_call_tools = is_call_tools_node(node)
                node = await agent_run.next(node)
                if was_call_tools:
                    task_manager.on_tool_iteration(agent_run.all_messages())
                task_manager.on_progress(agent_run.all_messages())

            # Final synchronization: ensure outstanding in-run work is applied before
            # returning result so history persists the latest summarized state.
            await task_manager.flush(agent_run.all_messages())

        return agent_run.result

    async def _astream_with_in_run_compaction(
        self, agent, prompt, message_history, deps, context
    ) -> AsyncGenerator[str, None]:
        """Stream via agent.iter so in-run compaction can happen mid-run."""
        from pydantic_graph import End

        processor = StreamProcessor()

        task_manager = self._build_in_run_compaction_task_manager(context)

        async with agent.iter(
            prompt,
            message_history=message_history,
            model_settings=self._model_settings,
            deps=deps,
        ) as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                if is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        async for message in stream.stream_responses():
                            for chunk in processor.process_message(message):
                                yield chunk

                was_call_tools = is_call_tools_node(node)

                node = await agent_run.next(node)
                if was_call_tools:
                    task_manager.on_tool_iteration(agent_run.all_messages())
                task_manager.on_progress(agent_run.all_messages())

            await task_manager.flush(agent_run.all_messages())

            for chunk in processor.close():
                yield chunk

            if self.history_manager and agent_run.result is not None:
                await self.history_manager.add_messages_async(
                    agent_run.result.new_messages(),
                    context=context,
                    apply_truncation=self._should_apply_post_run_compaction(),
                )

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
        self._run_count += 1
        input_str = input_data.get("input") or str(prompt)

        if self._should_use_in_run_compaction():
            loop = asyncio.new_event_loop()
            old_loop = None
            async_gen = None
            try:
                try:
                    old_loop = asyncio.get_event_loop()
                except RuntimeError:
                    old_loop = None
                asyncio.set_event_loop(loop)
                async_gen = self._astream_with_in_run_compaction(
                    agent,
                    prompt,
                    message_history,
                    deps,
                    input_str,
                )
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                    except StopAsyncIteration:
                        break
                    yield chunk
                return
            finally:
                if async_gen is not None:
                    try:
                        loop.run_until_complete(async_gen.aclose())
                    except Exception:
                        pass
                loop.close()
                asyncio.set_event_loop(old_loop)

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
            new_messages = result.new_messages()
            asyncio.run(
                self._apply_stream_in_run_compaction_if_needed(
                    new_messages,
                    context=input_str,
                )
            )
            self.history_manager.add_messages(
                new_messages,
                context=input_str,
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
        self._run_count += 1
        input_str = input_data.get("input") or str(prompt)

        if self._should_use_in_run_compaction():
            async for chunk in self._astream_with_in_run_compaction(
                agent,
                prompt,
                message_history,
                deps,
                input_str,
            ):
                yield chunk
            return

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
            if self.history_manager:
                new_messages = result.new_messages()
                await self._apply_stream_in_run_compaction_if_needed(
                    new_messages,
                    context=input_str,
                )
                await self.history_manager.add_messages_async(
                    new_messages,
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
        tool_schemas = get_agent_tool_schemas(self.agent)
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
