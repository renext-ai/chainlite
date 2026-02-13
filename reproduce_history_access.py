import os
import asyncio
from typing import Any, List
from chainlite import ChainLite, ChainLiteConfig
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
    TextPart,
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings


# Define a tool that tries to access history
async def history_tool(ctx: RunContext[Any], question: str) -> str:
    print(f"Tool called with question: {question}")

    # Check for history/messages in context
    if hasattr(ctx, "messages"):
        print(f"Messages found in context: {len(ctx.messages)}")
        for i, m in enumerate(ctx.messages):
            print(f"Message {i}: {type(m).__name__}")
        return "History found"
    else:
        print("No messages found in context.")
        return "History NOT found"


class MockModel(Model):
    def __init__(self, tool_name):
        self.tool_name = tool_name
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def system(self) -> str:
        return "mock-system"

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        print(f"MockModel request called with {len(messages)} messages")
        self.call_count += 1

        last_msg = messages[-1]
        content_str = ""

        if isinstance(last_msg, ModelRequest):
            for part in last_msg.parts:
                if isinstance(part, UserPromptPart):
                    content_str += str(part.content)

        if "Call" in content_str:
            print("Triggering tool call...")
            return ModelResponse(
                parts=[
                    TextPart(content="Sure, I'll call the tool."),
                    ToolCallPart(
                        tool_name=self.tool_name,
                        args={"question": "test question"},
                        tool_call_id="call_123",
                    ),
                ]
            )
        else:
            return ModelResponse(parts=[TextPart(content="Hello there.")])


async def main():
    # Setup ChainLite
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o", use_history=True, config_name="repro_test"
    )
    chain = ChainLite(config)

    # Register the tool
    chain.agent.tool(history_tool)

    # Inject Mock Model
    chain.agent.model = MockModel(tool_name="history_tool")

    # Run the chain
    print("Running chain...")
    try:
        # 1. Simple chat
        await chain.arun({"input": "Hello, my name is Aiden."})

        # 2. Trigger tool
        await chain.arun({"input": "Call the history_tool."})
    except Exception as e:
        print(f"Error running chain: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
    asyncio.run(main())
