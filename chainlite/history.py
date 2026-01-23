from typing import Optional, List
from loguru import logger
from pydantic import TypeAdapter
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    ToolReturnPart,
    SystemPromptPart,
)


class HistoryManager:
    """Manages conversation history with optional Redis persistence."""

    def __init__(
        self,
        session_id: str,
        redis_url: Optional[str] = None,
        max_messages: int = 45,
    ):
        self.session_id = session_id
        self.redis_url = redis_url
        self.max_messages = max_messages
        self._redis_client = None
        self._messages: List[ModelMessage] = []

        if redis_url:
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis client and load existing history."""
        try:
            import redis

            self._redis_client = redis.from_url(self.redis_url)
            self._load_from_redis()
            logger.info(f"History manager using Redis: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory storage.")
            self._redis_client = None

    def _load_from_redis(self) -> None:
        """Load message history from Redis."""
        if not self._redis_client:
            return
        try:
            key = f"chainlite:history:{self.session_id}"
            data = self._redis_client.get(key)
            if data:
                # Use TypeAdapter to validate and load the list of ModelMessages
                adapter = TypeAdapter(List[ModelMessage])
                self._messages = adapter.validate_json(data)

                # Apply smart truncation
                self._truncate_history()
        except Exception as e:
            logger.warning(f"Failed to load history from Redis: {e}")
            self._messages = []

    def _save_to_redis(self) -> None:
        """Save message history to Redis."""
        if not self._redis_client:
            return
        try:
            key = f"chainlite:history:{self.session_id}"
            # Trim before saving
            self._truncate_history()

            # Use TypeAdapter to serialize the list of ModelMessages
            adapter = TypeAdapter(List[ModelMessage])
            json_data = adapter.dump_json(self._messages)
            self._redis_client.set(key, json_data)
        except Exception as e:
            logger.warning(f"Failed to save history to Redis: {e}")

    @property
    def messages(self) -> List[ModelMessage]:
        """Get the current message history, with safety check for dangling tool calls."""
        if not self._messages:
            return []

        last_msg = self._messages[-1]

        if isinstance(last_msg, (ModelRequest, ModelResponse)):
            has_tool_call = False
            for part in last_msg.parts:

                if getattr(part, "part_kind", "") == "tool-call" or hasattr(
                    part, "tool_name"
                ):
                    has_tool_call = True
                    break

            if has_tool_call:
                logger.warning(
                    "⚠️ Detected dangling Tool Call in memory. Truncating history to fix API sequence."
                )

                return self._messages[:-1]

        return self._messages

    def add_messages(self, messages: List[ModelMessage]) -> None:
        """Add new messages to history."""
        self._messages.extend(messages)
        self._truncate_history()
        if self._redis_client:
            self._save_to_redis()

    def _truncate_history(self) -> None:
        """
        Smartly truncate history to max_messages, ensuring we start with a valid User/System message
        and do not break Tool sequences.
        """
        if len(self._messages) <= self.max_messages:
            return

        # 1. Initial naive cut
        # Take a bit more than max_messages to check for better cut points if needed,
        # but the request is to limit to max_messages.
        # Let's start by looking at the window of size max_messages from the end.
        start_index = len(self._messages) - self.max_messages
        candidate_messages = self._messages[start_index:]

        # 2. Find a safe starting point within this window
        safe_start_offset = 0
        for i, msg in enumerate(candidate_messages):
            # Check if it is a safe start:
            # - Should usually be a UserPrompt or SystemPrompt (in ModelRequest)
            # - Should NOT be a ToolReturn (which implies a preceding ToolCall)

            is_start_safe = False

            if isinstance(msg, ModelRequest):
                # Check parts
                # If it has UserPromptPart or SystemPromptPart, it's a good start.
                # If it ONLY has ToolReturnPart, it is NOT a good start.

                has_user_or_system = False
                has_tool_return = False

                for part in msg.parts:
                    if isinstance(part, (UserPromptPart, SystemPromptPart)):
                        has_user_or_system = True
                    if isinstance(part, ToolReturnPart):
                        has_tool_return = True

                if has_user_or_system and not has_tool_return:
                    is_start_safe = True

            # ModelResponse allows us to start? Not ideal, usually we want to start with User input.
            # But if we must... actually usually we want to start with User.

            if is_start_safe:
                safe_start_offset = i
                break

        # Apply the safe start
        # If we didn't find ANY safe start in the last max_messages, we might be in trouble.
        # But we default to the naive cut if we can't find a better one?
        # Or maybe we search BACKWARDS from the naive cut point?
        # Let's just take the best valid start we found in the tail.

        self._messages = candidate_messages[safe_start_offset:]

    def clear(self) -> None:
        """Clear the message history."""
        self._messages = []
        if self._redis_client:
            try:
                key = f"chainlite:history:{self.session_id}"
                self._redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to clear Redis history: {e}")
