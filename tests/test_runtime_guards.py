import os
from unittest.mock import MagicMock

import pytest

from chainlite import ChainLite, ChainLiteConfig
from chainlite.compaction import CompactionManager
from chainlite.truncators import SimpleTruncator


@pytest.mark.asyncio
async def test_run_raises_inside_running_event_loop():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    chain = ChainLite(ChainLiteConfig(llm_model_name="openai:gpt-4o", use_history=False))

    with pytest.raises(RuntimeError, match="ChainLite.run.*event loop"):
        chain.run({"input": "hello"})


@pytest.mark.asyncio
async def test_stream_raises_inside_running_event_loop():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    chain = ChainLite(ChainLiteConfig(llm_model_name="openai:gpt-4o", use_history=False))

    with pytest.raises(RuntimeError, match="ChainLite.stream.*event loop"):
        list(chain.stream({"input": "hello"}))


def test_should_use_in_run_handles_invalid_config_shape():
    manager = CompactionManager(
        history_manager=MagicMock(),
        config={"start_iter": 2, "max_concurrency": 4},  # missing start_run
        compactor=SimpleTruncator(threshold=10),
        post_run_start=1,
    )

    assert manager.should_use_in_run(run_count=10) is False
