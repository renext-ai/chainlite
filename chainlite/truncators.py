from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
import hashlib
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
)

if TYPE_CHECKING:
    from .core import ChainLite


class BaseHistoryTruncator:
    """Base class for history truncation strategies."""

    threshold: int = 5000

    def __init__(self, threshold: int = 5000):
        self.threshold = threshold
        # Track processed tool outputs to avoid repeated truncation/summarization.
        # key -> content hash
        self._processed_tool_outputs: Dict[str, str] = {}

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _tool_part_key(part: ToolReturnPart) -> str:
        # Prefer tool_call_id because it should uniquely identify a tool result.
        if part.tool_call_id:
            return f"id:{part.tool_call_id}"
        # Fallback key when tool_call_id is not present.
        timestamp = part.timestamp.isoformat() if part.timestamp else ""
        return f"fallback:{part.tool_name}|{timestamp}"

    def _should_process_tool_part(
        self, part: ToolReturnPart, min_length: Optional[int] = None
    ) -> bool:
        if not isinstance(part.content, str):
            return False
        if min_length is not None and len(part.content) <= min_length:
            return False

        key = self._tool_part_key(part)
        new_hash = self._content_hash(part.content)
        old_hash = self._processed_tool_outputs.get(key)

        if old_hash is None:
            return True

        # If tool_call_id exists and has been processed once, skip re-processing.
        if key.startswith("id:"):
            return False

        # For fallback keys, only skip identical content.
        return old_hash != new_hash

    def _mark_tool_part_processed(
        self, part: ToolReturnPart, original_content: Optional[str] = None
    ) -> None:
        if original_content is not None and not isinstance(original_content, str):
            return
        if original_content is None and not isinstance(part.content, str):
            return
        key = self._tool_part_key(part)
        content = original_content if original_content is not None else part.content
        self._processed_tool_outputs[key] = self._content_hash(content)

    async def atruncate(
        self, messages: List[ModelMessage], context: Optional[str] = None
    ) -> List[ModelMessage]:
        """Asynchronously truncate messages."""
        return messages

    def truncate(
        self, messages: List[ModelMessage], context: Optional[str] = None
    ) -> List[ModelMessage]:
        """Synchronously truncate messages."""
        return messages

    async def atruncate_part(
        self, part: ToolReturnPart, context: Optional[str] = None
    ) -> ToolReturnPart:
        """Asynchronously truncate a single ToolReturnPart."""
        return part

    def truncate_part(
        self, part: ToolReturnPart, context: Optional[str] = None
    ) -> ToolReturnPart:
        """Synchronously truncate a single ToolReturnPart."""
        return part


class SimpleTruncator(BaseHistoryTruncator):
    """Simple character-based truncation for Tool results."""

    def __init__(self, threshold: int = 5000):
        super().__init__(threshold=threshold)

    def _truncate_part(self, part: ToolReturnPart) -> ToolReturnPart:
        if self._should_process_tool_part(part, min_length=self.threshold):
            original_content = part.content
            truncated_content = (
                part.content[: self.threshold] + "\n\n... [Truncated due to length]"
            )
            truncated_part = ToolReturnPart(
                tool_name=part.tool_name,
                content=truncated_content,
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
            self._mark_tool_part_processed(
                truncated_part, original_content=original_content
            )
            return truncated_part
        return part

    def _process_messages(self, messages: List[ModelMessage]) -> List[ModelMessage]:
        new_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts = []
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        new_parts.append(self._truncate_part(part))
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(parts=new_parts))
            else:
                new_messages.append(msg)
        return new_messages

    def truncate_part(self, part, context=None):
        return self._truncate_part(part)

    async def atruncate_part(self, part, context=None):
        return self._truncate_part(part)

    def truncate(self, messages, context=None):
        return self._process_messages(messages)

    async def atruncate(self, messages, context=None):
        return self._process_messages(messages)


class AutoSummarizor(BaseHistoryTruncator):
    """Implicitly uses a specified agent (defaults to primary) to summarize tool results."""

    def __init__(
        self,
        model_name: str,
        threshold: int = 5000,
    ):
        super().__init__(threshold=threshold)
        self.model_name = model_name
        self._summarizer_agent: Optional["ChainLite"] = None
        # Track summarizer token usage
        self.summarizer_input_tokens: int = 0
        self.summarizer_output_tokens: int = 0
        self.summarizer_calls: int = 0

    def _get_summarizer(self) -> "ChainLite":
        if self._summarizer_agent is None:
            from .core import ChainLite, ChainLiteConfig

            config = ChainLiteConfig(
                llm_model_name=self.model_name,
                system_prompt="Extract key information, data, and conclusions from the tool output to help answer the user question. Keep it concise (under 500 words).",
                use_history=False,
            )
            self._summarizer_agent = ChainLite(config)
        return self._summarizer_agent

    async def _summarize_part_async(
        self, part: ToolReturnPart, context: Optional[str]
    ) -> ToolReturnPart:
        if self._should_process_tool_part(part, min_length=self.threshold):
            original_content = part.content
            summarizer = self._get_summarizer()
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            # Use agent.run() directly to capture usage
            result = await summarizer.agent.run(prompt)
            summary = result.output
            usage = result.usage()
            self.summarizer_input_tokens += usage.input_tokens
            self.summarizer_output_tokens += usage.output_tokens
            self.summarizer_calls += 1
            summarized_part = ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
            self._mark_tool_part_processed(
                summarized_part, original_content=original_content
            )
            return summarized_part
        return part

    def _summarize_part_sync(
        self, part: ToolReturnPart, context: Optional[str]
    ) -> ToolReturnPart:
        if self._should_process_tool_part(part, min_length=self.threshold):
            original_content = part.content
            summarizer = self._get_summarizer()
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            # Use agent.run_sync() directly to capture usage
            result = summarizer.agent.run_sync(prompt)
            summary = result.output
            usage = result.usage()
            self.summarizer_input_tokens += usage.input_tokens
            self.summarizer_output_tokens += usage.output_tokens
            self.summarizer_calls += 1
            summarized_part = ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
            self._mark_tool_part_processed(
                summarized_part, original_content=original_content
            )
            return summarized_part
        return part

    async def atruncate_part(self, part, context=None):
        return await self._summarize_part_async(part, context)

    def truncate_part(self, part, context=None):
        return self._summarize_part_sync(part, context)

    async def atruncate(self, messages, context=None):
        new_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts = []
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        # Await the async summarization
                        new_parts.append(
                            await self._summarize_part_async(part, context)
                        )
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(parts=new_parts))
            else:
                new_messages.append(msg)
        return new_messages

    def truncate(self, messages, context=None):
        new_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts = []
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        new_parts.append(self._summarize_part_sync(part, context))
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(parts=new_parts))
            else:
                new_messages.append(msg)
        return new_messages


class ChainLiteSummarizor(BaseHistoryTruncator):
    """Recursive ChainLite summarizer."""

    def __init__(
        self, config_or_path: Union[str, Dict[str, Any]], threshold: int = 5000
    ):
        super().__init__(threshold=threshold)
        from .core import ChainLite

        if isinstance(config_or_path, str):
            self.summarizer = ChainLite.load_config_from_yaml(config_or_path)
        else:
            from .core import ChainLite, ChainLiteConfig

            self.summarizer = ChainLite(ChainLiteConfig(**config_or_path))

        self.summarizer.config.use_history = False
        self.summarizer.history_manager = None

    async def _summarize_part_async(
        self, part: ToolReturnPart, context: Optional[str]
    ) -> ToolReturnPart:
        if self._should_process_tool_part(part, min_length=self.threshold):
            original_content = part.content
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            summary = await self.summarizer.arun({"input": prompt})
            summarized_part = ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
            self._mark_tool_part_processed(
                summarized_part, original_content=original_content
            )
            return summarized_part
        return part

    def _summarize_part_sync(
        self, part: ToolReturnPart, context: Optional[str]
    ) -> ToolReturnPart:
        if self._should_process_tool_part(part, min_length=self.threshold):
            original_content = part.content
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            # Ensure proper sync execution
            try:
                import asyncio

                # If there's an existing loop, use it? No, run() is strictly sync blocking.
                # Use Runner sync method if available, or create new loop if none exists.
                # However, pydantic-ai might not support run() from async context.
                # But here we are in truncate() which is sync.
                summary = self.summarizer.run({"input": prompt})
            except RuntimeError as e:
                # Fallback if somehow called from async context unexpectedly
                summary = f"Error summarizing: {e}"

            summarized_part = ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
            self._mark_tool_part_processed(
                summarized_part, original_content=original_content
            )
            return summarized_part
        return part

    async def atruncate_part(self, part, context=None):
        return await self._summarize_part_async(part, context)

    def truncate_part(self, part, context=None):
        return self._summarize_part_sync(part, context)

    async def atruncate(self, messages, context=None):
        new_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts = []
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        new_parts.append(
                            await self._summarize_part_async(part, context)
                        )
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(parts=new_parts))
            else:
                new_messages.append(msg)
        return new_messages

    def truncate(self, messages, context=None):
        new_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts = []
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        new_parts.append(self._summarize_part_sync(part, context))
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(parts=new_parts))
            else:
                new_messages.append(msg)
        return new_messages
