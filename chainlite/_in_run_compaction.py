import asyncio
from typing import Awaitable, Callable, List, Optional

from pydantic_ai.messages import ModelMessage


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
