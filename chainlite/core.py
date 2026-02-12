"""
Core ChainLite module using Pydantic AI.

This module provides a lightweight wrapper around Pydantic AI's Agent for building
LLM-powered applications with support for streaming, structured output, and
conversation history.
"""

from typing import Optional, Generator, AsyncGenerator, Dict, List, Any, Union
import re
import asyncio
import traceback
import json
import base64
import os
import mimetypes
from loguru import logger
import yaml
from pydantic import BaseModel, create_model, Field, TypeAdapter
import jinja2
from jinja2 import meta

from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings

from .config import ChainLiteConfig
from .provider import resolve_model_string
from .history import HistoryManager
from .streaming import StreamProcessor


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

        self.setup_chain()

    def _merge_dictionaries(self, dict_list: List[Dict]) -> Dict:
        """Merges a list of dictionaries into a single dictionary."""
        merged_dict = {}
        for single_dict in dict_list:
            merged_dict.update(single_dict)
        return merged_dict

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
            self.output_model = self.create_dynamic_pydantic_model(
                self._merge_dictionaries(self.config.output_parser)
            )

        # Truncator and History Setup
        truncator = None
        if self.config.history_truncator_config:
            from .truncators import SimpleTruncator, AutoSummarizor, ChainLiteSummarizor

            t_config = self.config.history_truncator_config
            mode = t_config.get("mode")
            threshold = t_config.get("truncation_threshold", 5000)

            if mode == "simple":
                truncator = SimpleTruncator(threshold=threshold)
            elif mode == "auto":
                truncator = AutoSummarizor(
                    threshold=threshold, model_name=self.config.llm_model_name
                )
            elif mode == "custom":
                path = t_config.get("summarizor_config_path")
                dict_cfg = t_config.get("summarizor_config_dict")
                if path and dict_cfg:
                    raise ValueError(
                        "Cannot specify both 'summarizor_config_path' and 'summarizor_config_dict'"
                    )
                if not path and not dict_cfg:
                    raise ValueError(
                        "Must specify either 'summarizor_config_path' or 'summarizor_config_dict' for custom truncator"
                    )
                truncator = ChainLiteSummarizor(
                    config_or_path=path or dict_cfg, threshold=threshold
                )

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

        # Configure model settings
        settings_dict = {}
        if self.config.model_settings:
            settings_dict.update(self.config.model_settings)

        if self.config.temperature is not None:
            settings_dict["temperature"] = self.config.temperature

        model_settings = None
        if settings_dict:
            model_settings = ModelSettings(**settings_dict)

        # Create the agent
        if self.output_model:
            self.agent = Agent(
                model_string,
                instructions=instructions,
                output_type=self.output_model,
                retries=self.config.max_retries or 3,
            )
        else:
            self.agent = Agent(
                model_string,
                instructions=instructions,
                retries=self.config.max_retries or 3,
            )

        self._model_settings = model_settings

    def _create_agent(self, instructions: Optional[str]) -> Agent:
        """Create an agent instance with the given instructions."""
        model_string = resolve_model_string(self.config.llm_model_name)

        # Collect tools from existing agent
        tools = []
        # Check for _function_toolset (pydantic-ai internal structure)
        if self.agent and hasattr(self.agent, "_function_toolset"):
            toolset = self.agent._function_toolset
            if hasattr(toolset, "tools"):
                tools.extend(toolset.tools.values())
        # Fallback/Older pydantic-ai version check
        elif self.agent and hasattr(self.agent, "_function_tools"):
            tools.extend(self.agent._function_tools.values())

        # Create new agent with tools
        if self.output_model:
            new_agent = Agent(
                model_string,
                instructions=instructions,
                output_type=self.output_model,
                retries=self.config.max_retries or 3,
                tools=tools,  # Pass existing tools
            )
        else:
            new_agent = Agent(
                model_string,
                instructions=instructions,
                retries=self.config.max_retries or 3,
                tools=tools,  # Pass existing tools
            )

        return new_agent

    def _build_instructions(self, context: Dict[str, Any] = None) -> Optional[str]:
        """Build system instructions from config, optionally rendering with context."""
        parts = []

        if self.config.system_prompt:
            system_prompt = self.config.system_prompt
            if context:
                # Check if system prompt has variables before attempting render
                # This avoids unnecessary processing if it's static
                if self.parse_input_variables_from_prompt(system_prompt):
                    try:
                        template = jinja2.Template(system_prompt)
                        system_prompt = template.render(**context)
                    except Exception as e:
                        logger.warning(f"Failed to render system_prompt: {e}")

            parts.append(system_prompt)

        if parts:
            return "\n\n".join(parts)
        return None

    async def _process_media_item(
        self, item: Any
    ) -> Union[ImageUrl, BinaryContent, str]:
        """
        Process a media item (URL, path, or key) into a Pydantic AI compatible format.
        """
        if isinstance(item, (ImageUrl, BinaryContent)):
            return item

        if isinstance(item, str):
            # 1. Base64 Check
            if item.startswith("data:"):
                try:
                    header, encoded = item.split(",", 1)
                    # header looks like "data:image/jpeg;base64"
                    media_type = header.split(";")[0].split(":")[1]
                    data = base64.b64decode(encoded)
                    return BinaryContent(data=data, media_type=media_type)
                except Exception as e:
                    logger.warning(f"Failed to process base64 string: {e}")
                    # Fallthrough to treating as string/url if failed?
                    # Probably safer to return as is or error. Let's return as is.
                    return item

            # 2. Remote URL Check
            if item.startswith(("http://", "https://")):
                return ImageUrl(item)

            # 3. Local File Check
            if os.path.exists(item):
                mime_type, _ = mimetypes.guess_type(item)
                if not mime_type:
                    mime_type = "image/jpeg"
                try:

                    def _read():
                        with open(item, "rb") as f:
                            return f.read()

                    file_data = await asyncio.to_thread(_read)
                    return BinaryContent(data=file_data, media_type=mime_type)
                except Exception as e:
                    logger.warning(f"Failed to read local file {item}: {e}")
                    raise e

            # 4. Fallback (Maybe it's just a non-existent file path or a weird URL)
            return ImageUrl(item)

        return str(item)

    async def _build_prompt(
        self, input_data: dict
    ) -> Union[str, list[Union[str, BinaryContent, ImageUrl]]]:
        """Build the user prompt from template and input data."""
        prompt_template = self.config.prompt or "{{ input }}"

        try:
            # Create a Jinja2 template and render it with the input data
            template = jinja2.Template(prompt_template)
            prompt_str = template.render(**input_data)
        except Exception as e:
            logger.warning(f"Error in prompt formatting: {e}")
            raise e

        content_list = [prompt_str]

        # 1. Check 'images' list input
        images = input_data.get("images")
        if images and isinstance(images, list):
            for img in images:
                content_list.append(await self._process_media_item(img))

        # 2. Check legacy 'image_url' input
        image_url = input_data.get("image_url")
        if image_url:
            content_list.append(await self._process_media_item(image_url))

        if len(content_list) > 1:
            return content_list

        return prompt_str

    def _get_message_history(self) -> Optional[List[ModelMessage]]:
        """Get message history if enabled."""
        if self.history_manager:
            return (
                self.history_manager.messages if self.history_manager.messages else None
            )
        return None

    def _update_history(
        self, new_messages: List[ModelMessage], context: Optional[str] = None
    ) -> None:
        """Update message history with new messages."""
        if self.history_manager:
            self.history_manager.add_messages(new_messages, context=context)

    async def _aupdate_history(
        self, new_messages: List[ModelMessage], context: Optional[str] = None
    ) -> None:
        """Update message history asynchronously."""
        if self.history_manager:
            await self.history_manager.add_messages_async(new_messages, context=context)

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
        message_history = self._get_message_history()
        return prompt, message_history

    def _process_run_result(
        self, result: Any, context: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Process agent run result: update history and dump output."""
        if self.history_manager:
            self._update_history(result.new_messages(), context=context)

        output = result.output
        if isinstance(output, BaseModel):
            return output.model_dump()
        return output

    async def _aprocess_run_result(
        self, result: Any, context: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Process agent run result asynchronously."""
        if self.history_manager:
            await self._aupdate_history(result.new_messages(), context=context)

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
        if self.config.system_prompt and self.parse_input_variables_from_prompt(
            self.config.system_prompt
        ):
            instructions = self._build_instructions(input_data)
            return self._create_agent(instructions)

        return self.agent

    def run(self, input_data: dict, deps: Any = None) -> Union[str, Dict[str, Any]]:
        """
        Executes the synchronous chat flow.

        Args:
            input_data: A dictionary containing the input data for the chat flow.
            deps: Optional dependencies to pass to the agent.

        Returns:
            The response from the language model, either as string or structured data.
        """
        prompt, message_history = asyncio.run(self._prepare_run(input_data))

        # Get agent (potentially dynamic)
        agent = asyncio.run(self._get_agent_for_run(input_data))

        max_retries = self.config.max_retries or 3
        current_try = 0

        while True:
            try:
                result = agent.run_sync(
                    prompt,
                    message_history=message_history,
                    model_settings=self._model_settings,
                    deps=deps,
                )
                # Capture the prompt string for context-aware truncation
                input_str = input_data.get("input") or str(prompt)
                return self._process_run_result(result, context=input_str)

            except Exception as e:
                current_try += 1
                self._handle_run_error(e, "run", prompt, input_data)

                if current_try >= max_retries:
                    logger.error("Max retries reached")
                    raise e

                logger.info(f"Retrying {current_try}/{max_retries}")

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
        prompt, message_history = await self._prepare_run(input_data)

        # Get agent (potentially dynamic)
        agent = await self._get_agent_for_run(input_data)

        max_retries = self.config.max_retries or 3
        current_try = 0

        while True:
            try:
                result = await agent.run(
                    prompt,
                    message_history=message_history,
                    model_settings=self._model_settings,
                    deps=deps,
                )
                input_str = input_data.get("input") or str(prompt)
                return await self._aprocess_run_result(result, context=input_str)

            except Exception as e:
                current_try += 1
                self._handle_run_error(e, "arun", prompt, input_data)

                if current_try >= max_retries:
                    logger.error("Max retries reached")
                    raise e

                logger.info(f"Retrying {current_try}/{max_retries}")
                await asyncio.sleep(1)

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
            self._update_history(result.new_messages())

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
                await self._aupdate_history(result.new_messages(), context=input_str)

    @property
    def chat_history_messages(self) -> Optional[List[ModelMessage]]:
        """Get the current chat history messages."""
        if self.history_manager:
            return self.history_manager.messages
        return None

    def _get_full_system_prompt(self) -> str:
        """Extract instructions and tool definitions from the agent."""
        if not self.agent:
            return ""

        lines = []
        # 1. Instructions
        if hasattr(self.agent, "_instructions") and self.agent._instructions:
            lines.append("## Instructions\n")
            for inst in self.agent._instructions:
                lines.append(f"{inst}\n")

        # 2. Tools
        if (
            hasattr(self.agent, "_function_toolset")
            and self.agent._function_toolset.tools
        ):
            if lines:
                lines.append("\n")
            lines.append("## Tools\n")
            for name, tool in self.agent._function_toolset.tools.items():
                lines.append(f"\n### Tool: {name}")
                if tool.description:
                    lines.append(f"**Description**: {tool.description}")

                # Build a complete tool schema for audit
                schema_data = {
                    "name": tool.name,
                    "description": tool.description,
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
    def parse_input_variables_from_prompt(text: str) -> List[str]:
        """Parses input variables from the prompt.

        Returns:
            A list of strings containing the input variables.
        """
        if not isinstance(text, str):
            return []

        env = jinja2.Environment()
        try:
            ast = env.parse(text)
            input_variables = meta.find_undeclared_variables(ast)
            return list(input_variables)
        except Exception as e:
            logger.error(f"Failed to parse input variables: {e}")
            return []

    @staticmethod
    def create_dynamic_pydantic_model(data: Dict[str, Any]) -> type[BaseModel]:
        """
        Creates a dynamic Pydantic model from a dictionary.

        Args:
            data: The dictionary from which to create the Pydantic model.

        Returns:
            A dynamically created Pydantic model class.
        """
        dynamic_model_fields = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Define fields with description
                dynamic_model_fields[key] = (str, Field(description=value))
            else:
                output_type = str
                output_description = None
                for v in value:
                    if v.get("type"):
                        output_type = ChainLite.get_type_from_string(v["type"])
                    if v.get("description"):
                        output_description = v["description"]

                dynamic_model_fields[key] = (
                    output_type,
                    (
                        Field(description=output_description)
                        if output_description
                        else ...
                    ),
                )

        # Dynamically create a model based on the keys and inferred types from the dictionary
        dynamic_model = create_model("DynamicOutput", **dynamic_model_fields)

        return dynamic_model

    @staticmethod
    def get_type_from_string(type_str: str) -> Any:
        """
        Gets the type from a available string representation.
        """
        available_types = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "Dict[str, Any]": Dict[str, Any],
            "Dict[str, str]": Dict[str, str],
            "List[str]": List[str],
            "List[Dict[str, Any]]": List[Dict[str, Any]],
        }
        try:
            return available_types[type_str]
        except KeyError:
            logger.error(f"Unknown type: {type_str}")
            return str

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
