import asyncio
import copy
from typing import Awaitable, Callable, Dict, List, Optional

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart


class InRunCompactionTaskManager:
    """Manage in-run compaction tasks across agent iterations."""

    def __init__(
        self,
        *,
        start_iter: int,
        apply_compaction: Callable[[List[ModelMessage]], Awaitable[None]],
    ) -> None:
        self._start_iter = start_iter
        self._apply_compaction = apply_compaction
        self._tool_iter_count = 0
        self._in_run_task: Optional[asyncio.Task] = None
        self._in_run_refresh_requested = False

    def _schedule_if_needed(self, messages: List[ModelMessage]) -> None:
        if self._in_run_task is None or self._in_run_task.done():
            self._in_run_task = asyncio.create_task(self._apply_compaction(messages))
            self._in_run_refresh_requested = False
        else:
            # Coalesce repeated triggers while one in-run task is in flight.
            self._in_run_refresh_requested = True

    def on_tool_iteration(self, messages: List[ModelMessage]) -> None:
        """Notify task manager after a call-tools iteration."""
        self._tool_iter_count += 1
        if self._tool_iter_count >= self._start_iter:
            self._schedule_if_needed(messages)

    def on_progress(self, messages: List[ModelMessage]) -> None:
        """Schedule a deferred refresh once the current in-run task completes."""
        if (
            self._in_run_task is not None
            and self._in_run_task.done()
            and self._in_run_refresh_requested
        ):
            self._schedule_if_needed(messages)

    async def flush(self, messages: List[ModelMessage]) -> None:
        """Ensure all pending in-run compaction work is finished."""
        if self._in_run_task is not None:
            await self._in_run_task
        if self._in_run_refresh_requested:
            await self._apply_compaction(messages)


class CompactionManager:
    """Encapsulate post-run/in-run compaction policies and execution."""

    def __init__(
        self,
        *,
        history_manager: Optional[object],
        config: Optional[dict],
        compactor: Optional[object],
        post_run_start: int,
    ) -> None:
        self.history_manager = history_manager
        self.config = config
        self.compactor = compactor
        self.post_run_start = post_run_start

    def should_apply_post_run(self, run_count: int) -> bool:
        return run_count >= self.post_run_start

    def should_use_in_run(self, run_count: int) -> bool:
        return (
            self.history_manager is not None
            and self.config is not None
            and self.compactor is not None
            and run_count >= self.config["start_run"]
        )

    def _count_tool_result_blocks(self, messages: List[ModelMessage]) -> int:
        count = 0
        for msg in messages:
            if isinstance(msg, ModelRequest) and any(
                isinstance(part, ToolReturnPart) for part in msg.parts
            ):
                count += 1
        return count

    def _should_include_latest_for_in_run(self) -> bool:
        return self.config is not None and int(self.config.get("start_iter", 2)) <= 1

    @staticmethod
    def _tool_part_key(part: ToolReturnPart) -> str:
        if part.tool_call_id:
            return f"id:{part.tool_call_id}"
        timestamp = part.timestamp.isoformat() if part.timestamp else ""
        return f"fallback:{part.tool_name}|{timestamp}"

    def _capture_raw_tool_results(
        self,
        messages: List[ModelMessage],
        raw_snapshot: Dict[str, str],
    ) -> None:
        """Capture original tool results before they are compacted in-place."""
        if self.compactor is None:
            return
        threshold = getattr(self.compactor, "threshold", 0)
        for msg in messages:
            if not isinstance(msg, ModelRequest):
                continue
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and isinstance(part.content, str)
                    and len(part.content) > threshold
                ):
                    key = self._tool_part_key(part)
                    raw_snapshot.setdefault(key, part.content)

    def restore_raw_messages_from_snapshot(
        self,
        messages: List[ModelMessage],
        raw_snapshot: Dict[str, str],
    ) -> List[ModelMessage]:
        """Restore compacted tool parts back to original raw content for audit logs."""
        restored = copy.deepcopy(messages)
        for msg in restored:
            if not isinstance(msg, ModelRequest):
                continue
            for part in msg.parts:
                if not isinstance(part, ToolReturnPart):
                    continue
                key = self._tool_part_key(part)
                original_content = raw_snapshot.get(key)
                if original_content is not None:
                    part.content = original_content
        return restored

    async def run_in_run_compaction(
        self,
        messages: List[ModelMessage],
        context: Optional[str],
        raw_snapshot: Optional[Dict[str, str]] = None,
    ) -> None:
        if (
            self.history_manager is None
            or self.config is None
            or self.compactor is None
        ):
            return

        if raw_snapshot is not None:
            self._capture_raw_tool_results(messages, raw_snapshot)

        await self.history_manager.apply_in_run_compaction_to_previous_tool_results(
            messages,
            compactor=self.compactor,
            context=context,
            max_concurrency=self.config.get("max_concurrency", 4),
            include_latest_tool_block=self._should_include_latest_for_in_run(),
        )

    def build_task_manager(
        self, context: Optional[str], raw_snapshot: Optional[Dict[str, str]] = None
    ) -> InRunCompactionTaskManager:
        if self.config is None:
            raise ValueError("In-run compaction config is not available")

        async def _apply_in_run_compaction(messages: List[ModelMessage]) -> None:
            await self.run_in_run_compaction(messages, context, raw_snapshot=raw_snapshot)

        return InRunCompactionTaskManager(
            start_iter=self.config["start_iter"],
            apply_compaction=_apply_in_run_compaction,
        )

    async def apply_stream_compaction_if_needed(
        self,
        messages: List[ModelMessage],
        context: Optional[str],
        run_count: int,
        raw_snapshot: Optional[Dict[str, str]] = None,
    ) -> None:
        if not self.should_use_in_run(run_count):
            return

        if self.config is None:
            return

        tool_blocks = self._count_tool_result_blocks(messages)
        if tool_blocks < self.config["start_iter"]:
            return

        await self.run_in_run_compaction(messages, context, raw_snapshot=raw_snapshot)
