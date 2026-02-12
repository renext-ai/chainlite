from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
)

if TYPE_CHECKING:
    from .core import ChainLite


class BaseHistoryTruncator:
    """Base class for history truncation strategies."""

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


class SimpleTruncator(BaseHistoryTruncator):
    """Simple character-based truncation for Tool results."""

    def __init__(self, threshold: int = 5000):
        self.threshold = threshold

    def _truncate_part(self, part: ToolReturnPart) -> ToolReturnPart:
        if isinstance(part.content, str) and len(part.content) > self.threshold:
            truncated_content = (
                part.content[: self.threshold] + "\n\n... [Truncated due to length]"
            )
            return ToolReturnPart(
                tool_name=part.tool_name,
                content=truncated_content,
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
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
        self.threshold = threshold
        self.model_name = model_name
        self._summarizer_agent: Optional["ChainLite"] = None

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
        if isinstance(part.content, str) and len(part.content) > self.threshold:
            summarizer = self._get_summarizer()
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            # Use arun for async context
            summary = await summarizer.arun({"input": prompt})
            return ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
        return part

    def _summarize_part_sync(
        self, part: ToolReturnPart, context: Optional[str]
    ) -> ToolReturnPart:
        if isinstance(part.content, str) and len(part.content) > self.threshold:
            summarizer = self._get_summarizer()
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            # Use run for sync context
            summary = summarizer.run({"input": prompt})
            return ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
        return part

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
        self.threshold = threshold
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
        if isinstance(part.content, str) and len(part.content) > self.threshold:
            prompt = f"User Question: {context}\n\nTool Output: {part.content}"
            summary = await self.summarizer.arun({"input": prompt})
            return ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
        return part

    def _summarize_part_sync(
        self, part: ToolReturnPart, context: Optional[str]
    ) -> ToolReturnPart:
        if isinstance(part.content, str) and len(part.content) > self.threshold:
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

            return ToolReturnPart(
                tool_name=part.tool_name,
                content=f"[Summarized Output]: {summary}",
                tool_call_id=part.tool_call_id,
                timestamp=part.timestamp,
            )
        return part

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
