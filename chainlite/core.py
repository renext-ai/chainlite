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
from loguru import logger
from pydantic import BaseModel
import jinja2

from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.messages import ModelMessage

from .compaction import CompactionManager, InRunCompactionConfig
from .config import ChainLiteConfig
from .config_loader import load_chainlite_from_yaml
from .provider import resolve_model_string
from .history import HistoryManager
from .streaming import StreamProcessor, StreamRunner
from ._factories import (
    build_compaction_components,
    build_model_settings,
    collect_agent_tools,
    create_agent_instance,
)
from .utils.media import build_prompt
from .utils.async_utils import ensure_no_running_loop
from .utils.output_model import (
    create_dynamic_pydantic_model,
    merge_dictionaries,
)
from .utils.prompts import parse_input_variables_from_prompt
from .adapters.pydantic_ai import (
    build_full_system_prompt,
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
        self.compaction_manager: Optional[CompactionManager] = None
        self.exception_retry = 0
        self._truncator = None
        self._post_run_compaction_start_run = 1
        self._in_run_compaction_config: Optional[InRunCompactionConfig] = None
        self._in_run_compactor = None
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
                self.history_manager.system_prompt = build_full_system_prompt(
                    self.agent
                )

        self.compaction_manager = CompactionManager(
            history_manager=self.history_manager,
            config=self._in_run_compaction_config,
            compactor=self._in_run_compactor,
            post_run_start=self._post_run_compaction_start_run,
        )

        # Build system instructions
        instructions = self._build_instructions()

        self._model_settings = build_model_settings(self.config)
        self.agent = create_agent_instance(
            model_string=model_string,
            instructions=instructions,
            output_model=self.output_model,
            retries=self.config.max_retries or 3,
        )

    def _sync_compaction_manager(self) -> None:
        """Keep compaction manager aligned with current runtime dependencies."""
        if self.compaction_manager is None:
            self.compaction_manager = CompactionManager(
                history_manager=self.history_manager,
                config=self._in_run_compaction_config,
                compactor=self._in_run_compactor,
                post_run_start=self._post_run_compaction_start_run,
            )
            return
        self.compaction_manager.history_manager = self.history_manager
        self.compaction_manager.config = self._in_run_compaction_config
        self.compaction_manager.compactor = self._in_run_compactor
        self.compaction_manager.post_run_start = self._post_run_compaction_start_run

    def _require_compaction_manager(self) -> CompactionManager:
        if self.compaction_manager is None:
            raise RuntimeError("Compaction manager is not initialized")
        return self.compaction_manager

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
        self._sync_compaction_manager()
        if self.agent is None:
            raise ValueError(
                "Agent is not set up. Please call `setup_chain` method before running."
            )

        # Ensure HistoryManager has the full system prompt (including dynamically added tools)
        if self.history_manager and not self.history_manager.system_prompt:
            self.history_manager.system_prompt = build_full_system_prompt(self.agent)

        prompt = await self._build_prompt(input_data)
        if self.history_manager and self.history_manager.messages:
            message_history = self.history_manager.messages
        else:
            message_history = None
        return prompt, message_history

    def _process_run_result(
        self,
        result: Any,
        context: Optional[str] = None,
        raw_messages: Optional[List[ModelMessage]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Process agent run result: update history and dump output."""
        compaction_manager = self._require_compaction_manager()
        if self.history_manager:
            self.history_manager.add_messages(
                result.new_messages(),
                context=context,
                apply_truncation=compaction_manager.should_apply_post_run(self._run_count),
                raw_messages=raw_messages,
            )

        output = result.output
        if isinstance(output, BaseModel):
            return output.model_dump()
        return output

    async def _aprocess_run_result(
        self,
        result: Any,
        context: Optional[str] = None,
        raw_messages: Optional[List[ModelMessage]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Process agent run result asynchronously."""
        compaction_manager = self._require_compaction_manager()
        if self.history_manager:
            await self.history_manager.add_messages_async(
                result.new_messages(),
                context=context,
                apply_truncation=compaction_manager.should_apply_post_run(self._run_count),
                raw_messages=raw_messages,
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

    async def _arun_with_in_run_compaction(
        self, agent, prompt, message_history, model_settings, deps, context
    ):
        """Run agent with in-run compaction of previous tool results."""
        from pydantic_graph import End

        compaction_manager = self._require_compaction_manager()
        raw_snapshot: dict[str, str] = {}
        task_manager = compaction_manager.build_task_manager(
            context, raw_snapshot=raw_snapshot
        )

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

        raw_messages = compaction_manager.restore_raw_messages_from_snapshot(
            agent_run.result.new_messages(),
            raw_snapshot,
        )
        return agent_run.result, raw_messages

    async def _astream_with_in_run_compaction(
        self, agent, prompt, message_history, deps, context
    ) -> AsyncGenerator[str, None]:
        """Stream via agent.iter so in-run compaction can happen mid-run."""
        from pydantic_graph import End

        compaction_manager = self._require_compaction_manager()
        processor = StreamProcessor()

        raw_snapshot: dict[str, str] = {}
        task_manager = compaction_manager.build_task_manager(
            context, raw_snapshot=raw_snapshot
        )

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
                raw_messages = (
                    compaction_manager.restore_raw_messages_from_snapshot(
                        agent_run.result.new_messages(),
                        raw_snapshot,
                    )
                )
                await self.history_manager.add_messages_async(
                    agent_run.result.new_messages(),
                    context=context,
                    apply_truncation=compaction_manager.should_apply_post_run(self._run_count),
                    raw_messages=raw_messages,
                )

    def _build_stream_runner(self, agent: Agent) -> StreamRunner:
        """Create a stream runner for a single prepared run."""
        compaction_manager = self._require_compaction_manager()
        return StreamRunner(
            agent=agent,
            model_settings=self._model_settings,
            history_manager=self.history_manager,
            should_apply_post_run_compaction=lambda: compaction_manager.should_apply_post_run(
                self._run_count
            ),
            apply_stream_in_run_compaction_if_needed=lambda messages, context: compaction_manager.apply_stream_compaction_if_needed(
                messages,
                context,
                self._run_count,
            ),
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
                compaction_manager = self._require_compaction_manager()
                raw_messages_override = None
                if compaction_manager.should_use_in_run(self._run_count):
                    result, raw_messages_override = (
                        await self._arun_with_in_run_compaction(
                            agent,
                            prompt,
                            message_history,
                            self._model_settings,
                            deps,
                            input_str,
                        )
                    )
                else:
                    result = await agent.run(
                        prompt,
                        message_history=message_history,
                        model_settings=self._model_settings,
                        deps=deps,
                    )

                if use_async_result_processing:
                    return await self._aprocess_run_result(
                        result,
                        context=input_str,
                        raw_messages=raw_messages_override,
                    )
                return self._process_run_result(
                    result,
                    context=input_str,
                    raw_messages=raw_messages_override,
                )

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
        ensure_no_running_loop("ChainLite.run")
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
        ensure_no_running_loop("ChainLite.stream")
        prompt, message_history = asyncio.run(self._prepare_run(input_data))

        # Get agent (potentially dynamic)
        agent = asyncio.run(self._get_agent_for_run(input_data))
        self._run_count += 1
        input_str = input_data.get("input") or str(prompt)
        stream_runner = self._build_stream_runner(agent)
        compaction_manager = self._require_compaction_manager()

        if compaction_manager.should_use_in_run(self._run_count):
            async_gen = self._astream_with_in_run_compaction(
                agent,
                prompt,
                message_history,
                deps,
                input_str,
            )
            for chunk in stream_runner.stream_sync_from_async_generator(async_gen):
                yield chunk
            return

        for chunk in stream_runner.stream_sync(
            prompt,
            message_history,
            deps,
            input_str,
        ):
            yield chunk

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
        stream_runner = self._build_stream_runner(agent)
        compaction_manager = self._require_compaction_manager()

        if compaction_manager.should_use_in_run(self._run_count):
            async for chunk in self._astream_with_in_run_compaction(
                agent,
                prompt,
                message_history,
                deps,
                input_str,
            ):
                yield chunk
            return

        async for chunk in stream_runner.stream_async(
            prompt,
            message_history,
            deps,
            input_str,
        ):
            yield chunk

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
        return load_chainlite_from_yaml(yaml_file_path, custom_configs)
