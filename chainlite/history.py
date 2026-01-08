from typing import Optional, List
from loguru import logger
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage


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

                # Trim to max messages
                if len(self._messages) > self.max_messages:
                    self._messages = self._messages[-self.max_messages :]
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
            if len(self._messages) > self.max_messages:
                self._messages = self._messages[-self.max_messages :]

            # Use TypeAdapter to serialize the list of ModelMessages
            adapter = TypeAdapter(List[ModelMessage])
            json_data = adapter.dump_json(self._messages)
            self._redis_client.set(key, json_data)
        except Exception as e:
            logger.warning(f"Failed to save history to Redis: {e}")

    @property
    def messages(self) -> List[ModelMessage]:
        """Get the current message history."""
        return self._messages

    def add_messages(self, messages: List[ModelMessage]) -> None:
        """Add new messages to history."""
        self._messages.extend(messages)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
        if self._redis_client:
            self._save_to_redis()

    def clear(self) -> None:
        """Clear the message history."""
        self._messages = []
        if self._redis_client:
            try:
                key = f"chainlite:history:{self.session_id}"
                self._redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to clear Redis history: {e}")
