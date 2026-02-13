import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
import os
from typing import Any
from chainlite.core import ChainLite
from chainlite.config import ChainLiteConfig
from dotenv import load_dotenv


class TestDepsPassthrough(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        os.environ.setdefault("OPENAI_API_KEY", "dummy")
        self.config = ChainLiteConfig(llm_model_name="openai:gpt-4o", use_history=False)
        self.chain_lite = ChainLite(self.config)

        # Mock the agent
        self.chain_lite.agent = MagicMock()
        self.chain_lite.agent.run_sync = MagicMock()
        self.chain_lite.agent.run = AsyncMock()

        async def async_iter():
            yield "chunk1"
            yield "chunk2"

        # Mock run_stream context manager
        self.mock_stream_result = MagicMock()
        # For sync stream, stream_text should return a sync iterable
        # For async stream, stream_text should return an async iterable
        # Since self.mock_stream_result is used for both, we need to make stream_text
        # a mock that can return different things or be called differently.
        # A simple way is to make stream_text itself a MagicMock and set its
        # side_effect or return_value in the specific test methods if needed,
        # or make it return a mock that can act as both.
        # For now, we'll set it up for the sync case and adjust in astream test.
        self.mock_stream_result.stream_text.return_value = [
            "chunk1",
            "chunk2",
        ]  # For sync stream
        # For async stream, we need to side_effect or handle it differently if possible,
        # or just make sure the test setup adjusts it for astream.
        # But wait, same mock object is used.
        # "stream_text" returns a Stream object in pydantic-ai which is iterable?
        # Actually in the code:
        # stream: "for chunk in result.stream_text(delta=True):"
        # astream: "async for chunk in result.stream_text(delta=True):"
        # Since I'm mocking the result object, I can't easily make .stream_text() return both a sync and async iterator
        # unless I separate them or use a side_effect that checks context (hard).
        # Better: create separate mocks for stream and astream tests, OR just overwrite it in the specific test.

        self.mock_stream_result.new_messages.return_value = []

        self.chain_lite.agent.run_stream_sync = MagicMock(
            return_value=self.mock_stream_result
        )

        self.chain_lite.agent.run_stream = MagicMock()
        self.chain_lite.agent.run_stream.return_value.__aenter__ = AsyncMock(
            return_value=self.mock_stream_result
        )
        self.chain_lite.agent.run_stream.return_value.__aexit__ = AsyncMock(
            return_value=None
        )

        # Mock dependencies
        self.deps = MagicMock()

    def test_run_passes_deps(self):
        self.chain_lite.agent.run.return_value.output = "response"
        self.chain_lite.agent.run.return_value.new_messages.return_value = []

        self.chain_lite.run({"input": "test"}, deps=self.deps)

        call_args = self.chain_lite.agent.run.call_args
        self.assertEqual(call_args.kwargs["deps"], self.deps)

    def test_arun_passes_deps(self):
        self.chain_lite.agent.run.return_value.output = "response"
        self.chain_lite.agent.run.return_value.new_messages.return_value = []

        asyncio.run(self.chain_lite.arun({"input": "test"}, deps=self.deps))

        call_args = self.chain_lite.agent.run.call_args
        self.assertEqual(call_args.kwargs["deps"], self.deps)

    def test_stream_passes_deps(self):
        # Consume the generator
        list(self.chain_lite.stream({"input": "test"}, deps=self.deps))

        call_args = self.chain_lite.agent.run_stream_sync.call_args
        self.assertEqual(call_args.kwargs["deps"], self.deps)

    def test_astream_passes_deps(self):
        async def async_iter():
            yield "chunk1"
            yield "chunk2"

        self.mock_stream_result.stream_text.return_value = async_iter()

        # Define helper at module level or just inline execution
        async def _helper():
            async for _ in self.chain_lite.astream({"input": "test"}, deps=self.deps):
                pass

        asyncio.run(_helper())

        call_args = self.chain_lite.agent.run_stream.call_args
        self.assertEqual(call_args.kwargs["deps"], self.deps)

    def test_stream_increments_run_count_and_applies_truncation_gate(self):
        self.chain_lite.history_manager = MagicMock()
        self.chain_lite._post_run_compaction_start_run = 1
        self.assertEqual(self.chain_lite._run_count, 0)

        list(self.chain_lite.stream({"input": "test"}, deps=self.deps))

        self.assertEqual(self.chain_lite._run_count, 1)
        self.chain_lite.history_manager.add_messages.assert_called_once()
        call_kwargs = self.chain_lite.history_manager.add_messages.call_args.kwargs
        self.assertTrue(call_kwargs["apply_truncation"])

    def test_astream_increments_run_count_and_applies_truncation_gate(self):
        async def async_iter():
            yield "chunk1"
            yield "chunk2"

        self.mock_stream_result.stream_text.return_value = async_iter()
        self.chain_lite.history_manager = MagicMock()
        self.chain_lite.history_manager.add_messages_async = AsyncMock()
        self.chain_lite._post_run_compaction_start_run = 1
        self.assertEqual(self.chain_lite._run_count, 0)

        async def _helper():
            async for _ in self.chain_lite.astream({"input": "test"}, deps=self.deps):
                pass

        asyncio.run(_helper())

        self.assertEqual(self.chain_lite._run_count, 1)
        self.chain_lite.history_manager.add_messages_async.assert_called_once()
        call_kwargs = self.chain_lite.history_manager.add_messages_async.call_args.kwargs
        self.assertTrue(call_kwargs["apply_truncation"])


if __name__ == "__main__":
    unittest.main()
