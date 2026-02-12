from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from loguru import logger
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    ToolReturnPart,
    SystemPromptPart,
)

if TYPE_CHECKING:
    from .core import ChainLite


from .trunctors import BaseHistoryTrunctor


class HistoryManager:
    """Manages conversation history with dual storage for context and raw logs."""

    def __init__(
        self,
        session_id: str,
        redis_url: Optional[str] = None,
        max_messages: int = 45,
        truncator: Optional[BaseHistoryTrunctor] = None,
    ):
        self.session_id = session_id
        self.redis_url = redis_url
        self.max_messages = max_messages
        self.truncator = truncator
        self._redis_client = None
        self._messages: List[ModelMessage] = []  # Context History (for LLM)
        self._raw_messages: List[ModelMessage] = []  # Raw History (for Audit)

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
            key_ctx = f"chainlite:history:{self.session_id}"
            key_raw = f"chainlite:history:raw:{self.session_id}"

            data_ctx = self._redis_client.get(key_ctx)
            if data_ctx:
                adapter = TypeAdapter(List[ModelMessage])
                self._messages = adapter.validate_json(data_ctx)
                self._truncate_history(self._messages)

            data_raw = self._redis_client.get(key_raw)
            if data_raw:
                adapter = TypeAdapter(List[ModelMessage])
                self._raw_messages = adapter.validate_json(data_raw)
        except Exception as e:
            logger.warning(f"Failed to load history from Redis: {e}")

    def _save_to_redis(self) -> None:
        """Save message history to Redis."""
        if not self._redis_client:
            return
        try:
            key_ctx = f"chainlite:history:{self.session_id}"
            key_raw = f"chainlite:history:raw:{self.session_id}"

            self._truncate_history(self._messages)

            adapter = TypeAdapter(List[ModelMessage])
            self._redis_client.set(key_ctx, adapter.dump_json(self._messages))
            self._redis_client.set(key_raw, adapter.dump_json(self._raw_messages))
        except Exception as e:
            logger.warning(f"Failed to save history to Redis: {e}")

    @property
    def messages(self) -> List[ModelMessage]:
        """Get the context message history (for LLM)."""
        return self._fix_dangling_tool_calls(self._messages)

    @property
    def raw_messages(self) -> List[ModelMessage]:
        """Get the raw message history (for Audit)."""
        return self._raw_messages

    def _fix_dangling_tool_calls(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        if not messages:
            return []
        last_msg = messages[-1]
        if isinstance(last_msg, (ModelRequest, ModelResponse)):
            has_tool_call = False
            for part in last_msg.parts:
                if getattr(part, "part_kind", "") == "tool-call" or hasattr(
                    part, "tool_name"
                ):
                    has_tool_call = True
                    break
            if has_tool_call:
                return messages[:-1]
        return messages

    def add_messages(
        self, messages: List[ModelMessage], context: Optional[str] = None
    ) -> None:
        """Add new messages to history (Synchronous)."""
        self._raw_messages.extend(messages)
        if self.truncator:
            processed_messages = self.truncator.truncate(messages, context=context)
            self._messages.extend(processed_messages)
        else:
            self._messages.extend(messages)

        self._truncate_history(self._messages)
        if self._redis_client:
            self._save_to_redis()

    async def add_messages_async(
        self, messages: List[ModelMessage], context: Optional[str] = None
    ) -> None:
        """Add new messages to history (Asynchronous)."""
        self._raw_messages.extend(messages)
        if self.truncator:
            processed_messages = await self.truncator.atruncate(
                messages, context=context
            )
            self._messages.extend(processed_messages)
        else:
            self._messages.extend(messages)

        self._truncate_history(self._messages)
        if self._redis_client:
            self._save_to_redis()

    def _truncate_history(self, messages_list: List[ModelMessage]) -> None:
        """Smartly truncate history list to max_messages."""
        if len(messages_list) <= self.max_messages:
            return

        start_index = len(messages_list) - self.max_messages
        candidate_messages = messages_list[start_index:]

        safe_start_offset = 0
        for i, msg in enumerate(candidate_messages):
            is_start_safe = False
            if isinstance(msg, ModelRequest):
                has_user_or_system = False
                has_tool_return = False
                for part in msg.parts:
                    if isinstance(part, (UserPromptPart, SystemPromptPart)):
                        has_user_or_system = True
                    if isinstance(part, ToolReturnPart):
                        has_tool_return = True
                if has_user_or_system and not has_tool_return:
                    is_start_safe = True

            if is_start_safe:
                safe_start_offset = i
                break

        # In-place update of the provided list is not possible like this,
        # but the class usage currently expects self._messages to be updated.
        if messages_list is self._messages:
            self._messages = candidate_messages[safe_start_offset:]

    def clear(self) -> None:
        """Clear the message history."""
        self._messages = []
        self._raw_messages = []
        if self._redis_client:
            try:
                key_ctx = f"chainlite:history:{self.session_id}"
                key_raw = f"chainlite:history:raw:{self.session_id}"
                self._redis_client.delete(key_ctx, key_raw)
            except Exception as e:
                logger.warning(f"Failed to clear Redis history: {e}")

    def export(
        self,
        export_type: str = "all",
        export_format: str = "json",
        output_dir: str = ".",
    ) -> List[str]:
        """Export conversation history.

        Args:
            export_type: 'all' (both), 'full' (raw), 'truncated' (context).
            export_format: 'json' or 'markdown'.
            output_dir: Directory to save the exported files.

        Returns:
            List of paths to exported files.
        """
        from pathlib import Path
        from datetime import datetime

        out_path = Path(output_dir) if output_dir else Path(".")
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = []

        to_export = []
        if export_type in ["all", "full"]:
            to_export.append(("raw", self._raw_messages))
        if export_type in ["all", "truncated"]:
            to_export.append(("truncated", self._messages))

        for name, msgs in to_export:
            filename = f"history_{self.session_id}_{name}_{timestamp}"
            if export_format == "json":
                file_path = out_path / f"{filename}.json"
                self._export_json(msgs, file_path)
            else:
                file_path = out_path / f"{filename}.md"
                self._export_markdown(msgs, file_path, name)
            exported_files.append(str(file_path))

        return exported_files

    def _export_json(
        self, messages: List[ModelMessage], path: Union[str, "Path"]
    ) -> None:
        adapter = TypeAdapter(List[ModelMessage])
        with open(path, "w") as f:
            f.write(adapter.dump_json(messages, indent=2).decode())
        logger.info(f"History exported to JSON: {path}")

    def _export_markdown(
        self, messages: List[ModelMessage], path: Union[str, "Path"], mode: str
    ) -> None:
        lines = [f"# Conversation History ({mode.capitalize()})\n"]
        lines.append(f"**Session ID**: {self.session_id}\n")
        lines.append("---\n")

        for msg in messages:
            role = "Unknown"
            if isinstance(msg, ModelRequest):
                role = "User/System (Request)"
            elif isinstance(msg, ModelResponse):
                role = "AI (Response)"

            lines.append(f"### {role}\n")
            for part in msg.parts:
                p_kind = getattr(part, "part_kind", "unknown")
                lines.append(f"**[{p_kind}]**\n")
                if hasattr(part, "content"):
                    content = str(part.content)
                    if p_kind == "tool-return":
                        lines.append(f"```text\n{content}\n```\n")
                    else:
                        lines.append(f"{content}\n")
                elif hasattr(part, "tool_name"):
                    t_name = getattr(part, "tool_name", "unknown")
                    args = getattr(part, "args", "{}")
                    lines.append(f"Tool Call: `{t_name}` with args: `{args}`\n")
            lines.append("---\n")

        with open(path, "w") as f:
            f.writelines(lines)
        logger.info(f"History exported to Markdown: {path}")
