"""
Streaming utilities for ChainLite.

Provides SSE (Server-Sent Events) stream generation for use with web frameworks.
"""

import asyncio
import json
import logging
import traceback
from typing import Any, AsyncGenerator, TYPE_CHECKING

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
        raise e
    except ValueError as e:
        logging.error("Value error in the streaming process: %s", e)
        raise e
    except TypeError as e:
        logging.error("Type error in the streaming process: %s", e)
        raise e
    except Exception as e:
        logging.error("Unexpected error in the streaming process: %s", e)
        traceback.print_exc()
        raise e
