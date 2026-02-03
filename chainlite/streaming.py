"""
Streaming utilities for ChainLite.

Provides SSE (Server-Sent Events) stream generation for use with web frameworks.
"""

import asyncio
import json
import logging
import traceback
from typing import Any, AsyncGenerator, Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ChainLite


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
