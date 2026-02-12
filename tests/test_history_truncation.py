import asyncio
from chainlite import ChainLite, ChainLiteConfig
from chainlite.truncators import SimpleTruncator
from pydantic_ai.messages import ModelRequest, ToolReturnPart
import os


async def test_simple_truncation():
    print("\n--- Testing Simple Truncation ---")
    truncator = SimpleTruncator(threshold=20)

    # Simulate a tool return
    msgs = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="big_tool",
                    content="This is a very long content that should be truncated by the simple truncator.",
                    tool_call_id="call_1",
                )
            ]
        )
    ]

    processed = truncator.truncate(msgs, context="What happened?")
    print(f"Context content: {processed[0].parts[0].content}")

    assert "... [Truncated due to length]" in processed[0].parts[0].content
    print("PASS: Simple truncation verified.")


async def test_auto_summarization():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy" or api_key == "your_openai_api_key_here":
        print("Skipping auto summarization test (no valid API key).")
        return

    print("\n--- Testing Auto Summarization ---")
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o-mini",
        use_history=True,
        history_truncator_config={"mode": "auto", "truncation_threshold": 50},
    )
    cl = ChainLite(config)

    long_content = """
    The GDP of the United States in 2023 was approximately $27 trillion. 
    The population is about 333 million people. 
    The capital is Washington, D.C. 
    The main industries are services, manufacturing, and technology.
    """

    msgs = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="info_tool", content=long_content, tool_call_id="call_2"
                )
            ]
        )
    ]

    print("Requesting auto-summarization (this calls LLM)...")
    await cl.history_manager.add_messages_async(msgs, context="Tell me about US GDP.")

    ctx_history = cl.history_manager.messages
    print(f"Summarized content: {ctx_history[0].parts[0].content}")

    assert "[Summarized Output]" in ctx_history[0].parts[0].content
    print("PASS: Auto summarization verified.")


if __name__ == "__main__":
    asyncio.run(test_simple_truncation())
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        asyncio.run(test_auto_summarization())
    else:
        print("Skipping auto summarization test (no API key).")
