import sys
import os
import json
import asyncio
from typing import List
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chainlite.core import ChainLite, HistoryManager, ChainLiteConfig


def test_serialization():
    print("Testing Serialization...")
    # Create some mock messages
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
    ]

    # Test dumping
    adapter = TypeAdapter(List[ModelMessage])
    json_bytes = adapter.dump_json(messages)
    print("Serialized:", json_bytes)

    # Test loading
    loaded_messages = adapter.validate_json(json_bytes)
    assert len(loaded_messages) == 2
    assert isinstance(loaded_messages[0], ModelRequest)
    assert loaded_messages[0].parts[0].content == "Hello"
    print("Serialization test passed!")


def test_prompt_formatting():
    print("\nTesting Prompt Formatting...")
    config = ChainLiteConfig(
        prompt="Hello {name}, welcome to {place}!", llm_model_name="openai:test-model"
    )
    chain = ChainLite(config)

    # Test successful formatting
    input_data = {"name": "Alice", "place": "Wonderland"}
    prompt = chain._build_prompt(input_data)
    assert prompt == "Hello Alice, welcome to Wonderland!"
    print("Prompt formatting passed!")

    # Test missing key (should raise KeyError)
    try:
        chain._build_prompt({"name": "Bob"})
        print("ERROR: Should have raised KeyError")
    except KeyError:
        print("Correctly raised KeyError for missing key")


async def test_retry_logic():
    print("\nTesting Retry Logic...")
    config = ChainLiteConfig(max_retries=3, llm_model_name="openai:test-model")
    chain = ChainLite(config)

    # Mock agent
    mock_agent = MagicMock()
    chain.agent = mock_agent

    # Sync Run Retry
    # First 2 calls raise exception, 3rd succeeds
    mock_agent.run_sync.side_effect = [
        Exception("Fail 1"),
        Exception("Fail 2"),
        MagicMock(output="Success"),
    ]

    res = chain.run({"input": "test"})
    assert res == "Success"
    assert mock_agent.run_sync.call_count == 3
    print("Sync retry passed!")

    # Async Run Retry
    # First 2 calls raise exception, 3rd succeeds
    mock_agent.run = AsyncMock(
        side_effect=[
            Exception("Fail 1"),
            Exception("Fail 2"),
            MagicMock(output="Success"),
        ]
    )

    res = await chain.arun({"input": "test"})
    assert res == "Success"
    assert mock_agent.run.call_count == 3
    print("Async retry passed!")


if __name__ == "__main__":
    test_serialization()
    test_prompt_formatting()
    asyncio.run(test_retry_logic())
