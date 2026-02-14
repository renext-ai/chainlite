"""Async-related helper utilities."""

import asyncio


def ensure_no_running_loop(api_name: str) -> None:
    """Raise a clear error for sync APIs called inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{api_name} cannot be used while an event loop is running. "
        "Use the async API instead."
    )
