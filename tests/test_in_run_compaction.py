"""Tests for in-run compaction feature."""
import os
import asyncio
from typing import Any, List

from chainlite import ChainLite, ChainLiteConfig
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    TextPart,
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings


class MultiToolMockModel(Model):
    """Mock model that triggers N sequential tool calls then returns a final answer."""

    def __init__(self, tool_name: str, num_calls: int = 3):
        self.tool_name = tool_name
        self.num_calls = num_calls
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-multi-tool"

    @property
    def system(self) -> str:
        return "mock"

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.call_count += 1

        if self.call_count <= self.num_calls:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=self.tool_name,
                        args={"step": self.call_count},
                        tool_call_id=f"call_{self.call_count}",
                    ),
                ]
            )
        else:
            return ModelResponse(
                parts=[TextPart(content="All tools completed.")]
            )


def make_long_content(step: int) -> str:
    """Generate tool output that exceeds the threshold."""
    return f"STEP_{step}_START " + ("x" * 200) + f" STEP_{step}_END"


def collect_tool_returns(messages: List[ModelMessage]) -> List[ToolReturnPart]:
    tool_returns = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    tool_returns.append(part)
    return tool_returns


async def test_in_run_compaction_basic():
    """Test that in-run compaction truncates previous tool results during a run."""
    print("\n--- Test: In-Run Compaction Basic ---")

    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        session_id="test_in_run_basic",
        history_truncator_config={
            "post_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 5000,
            },
            "in_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 50,
                "start_iter": 2,
                "start_run": 1,
            },
        },
    )
    chain = ChainLite(config)

    # Register a tool that returns long content
    @chain.agent.tool_plain
    def big_tool(step: int) -> str:
        """A tool that returns large content."""
        return make_long_content(step)

    # Inject mock model that calls big_tool 3 times
    mock = MultiToolMockModel(tool_name="big_tool", num_calls=3)
    chain.agent.model = mock

    result = await chain.arun({"input": "Run all tools."})
    print(f"Result: {result}")
    print(f"Mock model was called {mock.call_count} times")

    # Check the history messages for truncation
    assert chain.history_manager is not None
    raw_msgs = chain.history_manager.raw_messages
    ctx_msgs = chain.history_manager.messages

    raw_tool_returns = collect_tool_returns(raw_msgs)
    ctx_tool_returns = collect_tool_returns(ctx_msgs)

    print(f"Found {len(raw_tool_returns)} tool returns in raw history")
    print(f"Found {len(ctx_tool_returns)} tool returns in context history")

    assert len(raw_tool_returns) == 3, f"Expected 3 raw tool returns, got {len(raw_tool_returns)}"
    assert len(ctx_tool_returns) == 3, f"Expected 3 context tool returns, got {len(ctx_tool_returns)}"

    # Raw history should preserve original tool outputs for audit.
    assert "... [Truncated due to length]" not in raw_tool_returns[0].content
    assert "... [Truncated due to length]" not in raw_tool_returns[1].content

    # Context history should reflect in-run compaction.
    assert "... [Truncated due to length]" in ctx_tool_returns[0].content
    assert "... [Truncated due to length]" in ctx_tool_returns[1].content
    assert "STEP_3_END" in ctx_tool_returns[2].content

    print("PASS: In-run compaction basic test")


async def test_in_run_start_run_delay():
    """Test that in-run compaction only activates after start_run threshold."""
    print("\n--- Test: In-Run Start Run Delay ---")

    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        session_id="test_in_run_delay",
        history_truncator_config={
            "post_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 5000,
            },
            "in_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 50,
                "start_iter": 2,
                "start_run": 3,  # Only activate on 3rd run
            },
        },
    )
    chain = ChainLite(config)

    @chain.agent.tool_plain
    def big_tool(step: int) -> str:
        """A tool that returns large content."""
        return make_long_content(step)

    # Run 1: in-run compaction should NOT be active (run_count=1 < start_run=3)
    mock1 = MultiToolMockModel(tool_name="big_tool", num_calls=2)
    chain.agent.model = mock1
    await chain.arun({"input": "Run 1"})

    assert chain._run_count == 1
    print(f"Run 1 complete, run_count={chain._run_count}")

    # Run 2: in-run compaction should NOT be active
    mock2 = MultiToolMockModel(tool_name="big_tool", num_calls=2)
    chain.agent.model = mock2
    await chain.arun({"input": "Run 2"})

    assert chain._run_count == 2
    print(f"Run 2 complete, run_count={chain._run_count}")

    # Collect tool returns from runs 1-2 (should NOT be in-run truncated)
    tool_returns_before = collect_tool_returns(chain.history_manager._raw_messages)

    for i, tr in enumerate(tool_returns_before):
        # None should be in-run truncated (post-run truncation threshold is 5000, content ~220)
        assert "... [Truncated due to length]" not in tr.content, \
            f"Tool return {i} should NOT be truncated before start_run threshold"

    print("Runs 1-2: no in-run truncation (correct)")

    # Run 3: in-run compaction SHOULD be active (run_count=3 >= start_run=3)
    chain.history_manager.clear()  # Clear history to isolate test
    mock3 = MultiToolMockModel(tool_name="big_tool", num_calls=3)
    chain.agent.model = mock3
    await chain.arun({"input": "Run 3"})

    assert chain._run_count == 3
    print(f"Run 3 complete, run_count={chain._run_count}")

    tool_returns_after = collect_tool_returns(chain.history_manager._raw_messages)
    ctx_tool_returns_after = collect_tool_returns(chain.history_manager.messages)

    # Raw history remains untruncated even when in-run compaction is active.
    assert len(tool_returns_after) == 3
    assert "... [Truncated due to length]" not in tool_returns_after[0].content
    # Context history still shows in-run compaction effect.
    assert "... [Truncated due to length]" in ctx_tool_returns_after[0].content

    print("Run 3: in-run truncation active (correct)")
    print("PASS: In-run start_run delay test")


async def test_in_run_start_iter_delay():
    """Test that in-run compaction respects start_iter within a single run."""
    print("\n--- Test: In-Run Start Iter Delay ---")

    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        session_id="test_in_run_iter",
        history_truncator_config={
            "post_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 5000,
            },
            "in_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 50,
                "start_iter": 3,  # Only start at 3rd tool iteration
                "start_run": 1,
            },
        },
    )
    chain = ChainLite(config)

    @chain.agent.tool_plain
    def big_tool(step: int) -> str:
        """A tool that returns large content."""
        return make_long_content(step)

    mock = MultiToolMockModel(tool_name="big_tool", num_calls=4)
    chain.agent.model = mock
    await chain.arun({"input": "Run tools"})

    raw_tool_returns = collect_tool_returns(chain.history_manager._raw_messages)
    ctx_tool_returns = collect_tool_returns(chain.history_manager.messages)

    assert len(raw_tool_returns) == 4
    assert len(ctx_tool_returns) == 4

    # start_iter=3 means:
    # - iter 1 (tool #1): no compaction (iter_count=1 < 3)
    # - iter 2 (tool #2): no compaction (iter_count=2 < 3)
    # - iter 3 (tool #3): compaction kicks in, truncate tool #1 and #2
    # - iter 4 (tool #4): compaction, truncate tool #1, #2, #3
    # Raw history should preserve original tool outputs.
    assert "... [Truncated due to length]" not in raw_tool_returns[0].content
    assert "... [Truncated due to length]" not in raw_tool_returns[1].content
    assert "... [Truncated due to length]" not in raw_tool_returns[2].content
    assert "STEP_4_END" in raw_tool_returns[3].content
    # Context history should show compaction results.
    assert "... [Truncated due to length]" in ctx_tool_returns[0].content
    assert "... [Truncated due to length]" in ctx_tool_returns[1].content
    assert "... [Truncated due to length]" in ctx_tool_returns[2].content
    assert "STEP_4_END" in ctx_tool_returns[3].content

    print("PASS: In-run start_iter delay test")


async def test_in_run_start_iter_one_compacts_first_tool_output():
    """start_iter=1 should allow compacting the first tool output in-run."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        session_id="test_in_run_iter_one",
        history_truncator_config={
            "post_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 5000,
            },
            "in_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 50,
                "start_iter": 1,
                "start_run": 1,
            },
        },
    )
    chain = ChainLite(config)

    @chain.agent.tool_plain
    def big_tool(step: int) -> str:
        return make_long_content(step)

    mock = MultiToolMockModel(tool_name="big_tool", num_calls=1)
    chain.agent.model = mock
    await chain.arun({"input": "Run one tool."})

    raw_tool_returns = collect_tool_returns(chain.history_manager._raw_messages)
    ctx_tool_returns = collect_tool_returns(chain.history_manager.messages)

    assert len(raw_tool_returns) == 1
    assert len(ctx_tool_returns) == 1
    assert "... [Truncated due to length]" not in raw_tool_returns[0].content
    assert "... [Truncated due to length]" in ctx_tool_returns[0].content


async def test_in_run_start_iter_two_does_not_compact_first_tool_output():
    """start_iter=2 should keep single first tool output un-compacted in-run."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        session_id="test_in_run_iter_two",
        history_truncator_config={
            "post_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 5000,
            },
            "in_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 50,
                "start_iter": 2,
                "start_run": 1,
            },
        },
    )
    chain = ChainLite(config)

    @chain.agent.tool_plain
    def big_tool(step: int) -> str:
        return make_long_content(step)

    mock = MultiToolMockModel(tool_name="big_tool", num_calls=1)
    chain.agent.model = mock
    await chain.arun({"input": "Run one tool."})

    raw_tool_returns = collect_tool_returns(chain.history_manager._raw_messages)
    ctx_tool_returns = collect_tool_returns(chain.history_manager.messages)

    assert len(raw_tool_returns) == 1
    assert len(ctx_tool_returns) == 1
    assert "... [Truncated due to length]" not in raw_tool_returns[0].content
    assert "... [Truncated due to length]" not in ctx_tool_returns[0].content


async def test_in_run_compaction_keeps_raw_audit_unmodified():
    """Raw audit history should retain original tool output while context is compacted."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        session_id="test_in_run_raw_audit",
        history_truncator_config={
            "post_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 5000,
            },
            "in_run_compaction": {
                "mode": "simple",
                "truncation_threshold": 50,
                "start_iter": 2,
                "start_run": 1,
            },
        },
    )
    chain = ChainLite(config)

    @chain.agent.tool_plain
    def big_tool(step: int) -> str:
        return make_long_content(step)

    mock = MultiToolMockModel(tool_name="big_tool", num_calls=3)
    chain.agent.model = mock
    await chain.arun({"input": "Run all tools."})

    raw_tool_returns = collect_tool_returns(chain.history_manager.raw_messages)
    ctx_tool_returns = collect_tool_returns(chain.history_manager.messages)

    assert len(raw_tool_returns) >= 2
    assert len(ctx_tool_returns) >= 2
    assert "... [Truncated due to length]" not in raw_tool_returns[0].content
    assert "... [Truncated due to length]" in ctx_tool_returns[0].content


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"

    asyncio.run(test_in_run_compaction_basic())
    asyncio.run(test_in_run_start_run_delay())
    asyncio.run(test_in_run_start_iter_delay())
    asyncio.run(test_in_run_start_iter_one_compacts_first_tool_output())
    asyncio.run(test_in_run_start_iter_two_does_not_compact_first_tool_output())
    asyncio.run(test_in_run_compaction_keeps_raw_audit_unmodified())
    print("\n=== All in-run compaction tests passed ===")
