"""
Streaming utilities for ChainLite.

Provides SSE (Server-Sent Events) stream generation for use with web frameworks.
"""

import asyncio
import json
import logging
import traceback
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Optional,
    Protocol,
    TYPE_CHECKING,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from .core import ChainLite
    from .history import HistoryManager


def _ensure_no_running_loop(api_name: str) -> None:
    """Raise a clear error for sync APIs called inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{api_name} cannot be used while an event loop is running. "
        "Use the async API instead."
    )


class AgentStreamLike(Protocol):
    def run_stream_sync(
        self,
        prompt: Any,
        message_history: Optional[list[ModelMessage]],
        model_settings: Optional[ModelSettings],
        deps: Any,
    ) -> Any: ...

    def run_stream(
        self,
        prompt: Any,
        message_history: Optional[list[ModelMessage]],
        model_settings: Optional[ModelSettings],
        deps: Any,
    ) -> Any: ...


async def stream_sse(
    chain: "ChainLite",
    payload: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Generate events for an SSE (Server-Sent Events) stream.

    Args:
        chain: The ChainLite instance with a configured agent.
        payload: The payload for the agent execution.

    Yields:
        str: JSON-formatted output of each response chunk, formatted for SSE.
    """
    try:
        async for output in chain.astream(payload):
            # Convert the output to JSON
            json_output = json.dumps(output)

            # Yield each JSON message in the correct SSE format
            yield f"data: {json_output}\n\n"
    except asyncio.TimeoutError as e:
        logging.error("Timeout occurred in the streaming process: %s", e)
        raise
    except ValueError as e:
        logging.error("Value error in the streaming process: %s", e)
        raise
    except TypeError as e:
        logging.error("Type error in the streaming process: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error in the streaming process: %s", e)
        traceback.print_exc()
        raise


class StreamProcessor:
    """
    Helper class to process streaming messages, tracking parts and handling tags.
    """

    def __init__(self):
        # Track which part index we are currently processing
        self.current_part_index = 0
        # Track the offset within the current part
        self.current_part_offset = 0
        # Track the currently active XML tag (e.g., 'thinking')
        self.active_tag = None

    def process_message(self, message: Any) -> Generator[str, None, None]:
        """
        Process a single ModelMessage (snapshot) and yield incremental text chunks.
        Handling Part transitions and injecting tags (e.g., <thinking>).
        """
        if not message.parts:
            return

        # Iterate through parts starting from the last processed index.
        # This handles cases where a snapshot contains the tail of an old part
        # and the beginning of a new part.
        for i in range(self.current_part_index, len(message.parts)):
            part = message.parts[i]

            # Get content (some parts might not have content, handle accordingly)
            content = getattr(part, "content", "") or ""

            # Determine where to slice this part:
            # - If it's the current part (i == current_part_index), accept from offset.
            # - If it's a new part (i > current_part_index), start from 0.
            start_pos = self.current_part_offset if i == self.current_part_index else 0

            # Only output if length has increased
            if len(content) > start_pos:
                delta = content[start_pos:]

                # Start of a new part: Handle headers/tags
                if start_pos == 0:
                    # Close previous tag if exists
                    if self.active_tag:
                        yield f"</{self.active_tag}>\n"
                        self.active_tag = None

                    # Get part kind
                    part_kind = getattr(part, "part_kind", "part")
                    pk_upper = part_kind.upper()

                    if pk_upper != "TEXT":
                        yield f"<{pk_upper}>"
                        self.active_tag = pk_upper

                yield delta

                # Update pointers
                if i == self.current_part_index:
                    # Same part, update offset
                    self.current_part_offset = len(content)
                else:
                    # Moved to next part, update index and reset offset
                    self.current_part_index = i
                    self.current_part_offset = len(content)

    def close(self) -> Generator[str, None, None]:
        """Close any remaining tags."""
        if self.active_tag:
            yield f"</{self.active_tag}>"
            self.active_tag = None


class StreamRunner:
    """Orchestrate sync/async streaming with StreamProcessor."""

    def __init__(
        self,
        *,
        agent: AgentStreamLike,
        model_settings: Optional[ModelSettings],
        history_manager: Optional["HistoryManager"],
        should_apply_post_run_compaction: Callable[[], bool],
        apply_stream_in_run_compaction_if_needed: Callable[
            [list[ModelMessage], Optional[str]], Awaitable[None]
        ],
    ) -> None:
        self._agent = agent
        self._model_settings = model_settings
        self._history_manager = history_manager
        self._should_apply_post_run_compaction = should_apply_post_run_compaction
        self._apply_stream_in_run_compaction_if_needed = (
            apply_stream_in_run_compaction_if_needed
        )

    def _iter_processor_chunks_sync(self, result: Any) -> Generator[str, None, None]:
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

        for chunk in processor.close():
            yield chunk

    async def _iter_processor_chunks_async(
        self, result: Any
    ) -> AsyncGenerator[str, None]:
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

        for chunk in processor.close():
            yield chunk

    def _persist_history_sync(self, result: Any, context: Optional[str]) -> None:
        if not self._history_manager:
            return

        new_messages = result.new_messages()
        _ensure_no_running_loop("StreamRunner._persist_history_sync")
        asyncio.run(self._apply_stream_in_run_compaction_if_needed(new_messages, context))
        self._history_manager.add_messages(
            new_messages,
            context=context,
            apply_truncation=self._should_apply_post_run_compaction(),
        )

    async def _persist_history_async(self, result: Any, context: Optional[str]) -> None:
        if not self._history_manager:
            return

        new_messages = result.new_messages()
        await self._apply_stream_in_run_compaction_if_needed(new_messages, context)
        await self._history_manager.add_messages_async(
            new_messages,
            context=context,
            apply_truncation=self._should_apply_post_run_compaction(),
        )

    def stream_sync(
        self,
        prompt: Any,
        message_history: Optional[list[ModelMessage]],
        deps: Any,
        context: Optional[str],
    ) -> Generator[str, None, None]:
        result = self._agent.run_stream_sync(
            prompt,
            message_history=message_history,
            model_settings=self._model_settings,
            deps=deps,
        )
        for chunk in self._iter_processor_chunks_sync(result):
            yield chunk
        self._persist_history_sync(result, context)

    async def stream_async(
        self,
        prompt: Any,
        message_history: Optional[list[ModelMessage]],
        deps: Any,
        context: Optional[str],
    ) -> AsyncGenerator[str, None]:
        async with self._agent.run_stream(
            prompt,
            message_history=message_history,
            model_settings=self._model_settings,
            deps=deps,
        ) as result:
            async for chunk in self._iter_processor_chunks_async(result):
                yield chunk
            await self._persist_history_async(result, context)

    def stream_sync_from_async_generator(
        self, async_gen: AsyncGenerator[str, None]
    ) -> Generator[str, None, None]:
        _ensure_no_running_loop("StreamRunner.stream_sync_from_async_generator")
        loop = asyncio.new_event_loop()
        old_loop = None
        try:
            try:
                old_loop = asyncio.get_event_loop()
            except RuntimeError:
                old_loop = None
            asyncio.set_event_loop(loop)
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
                yield chunk
        finally:
            try:
                loop.run_until_complete(async_gen.aclose())
            except Exception:
                pass
            loop.close()
            asyncio.set_event_loop(old_loop)
