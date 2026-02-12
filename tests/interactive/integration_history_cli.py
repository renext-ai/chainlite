import asyncio
import os
import argparse
from loguru import logger
from dotenv import load_dotenv
from chainlite import ChainLite
from pydantic_ai.messages import (
    ModelResponse,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)

# Load environment variables
load_dotenv()
from pydantic_ai import RunContext


async def integration_history_cli(config_path: str):
    """
    Integration test CLI for observing history truncation and tool interactions.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    try:
        # Load chain
        chain = ChainLite.load_config_from_yaml(config_path)

        # Register tools
        @chain.agent.tool
        def generate_long_text(ctx: RunContext, length: int = 500) -> str:
            """Generate a long string of text."""
            return "This is a repeated segment of text for testing truncation. " * (
                length // 10
            )

        @chain.agent.tool
        def simulate_data_fetch(ctx: RunContext, items: int = 50) -> str:
            """Simulate fetching a large dataset."""
            return "\n".join(
                [f"Item {i}: Some detailed data about item {i}" for i in range(items)]
            )

        logger.info(f"Initialized agent with tools: {chain.config.config_name}")
        logger.info("Tools registered: generate_long_text, simulate_data_fetch")

        print("\n" + "=" * 60)
        print("  ChainLite History Truncation Integration Test CLI")
        print("=" * 60)
        print("Commands: 'exit', 'quit', 'clear', 'export', 'history'")
        print("-" * 60 + "\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                if chain.history_manager:
                    chain.history_manager.clear()
                    print("\n[History cleared]\n")
                continue

            if user_input.lower() == "export":
                if chain.history_manager:
                    paths = chain.history_manager.export(
                        export_type="all", export_format="markdown"
                    )
                    print(f"\n[History exported to: {paths}]\n")
                continue

            if user_input.lower() == "history":
                if chain.history_manager:
                    print(
                        f"\n--- Context History Count: {len(chain.history_manager._messages)} ---"
                    )
                    print(
                        f"--- Raw History Count: {len(chain.history_manager._raw_messages)} ---"
                    )
                    if chain.history_manager._messages:
                        last_ctx = chain.history_manager._messages[-1]
                        print(f"Last Context Msg Sample: {str(last_ctx)[:200]}...")
                continue

            if not user_input:
                continue

            print("\nAgent: ", end="", flush=True)
            try:
                history_start_index = (
                    len(chain.history_manager.raw_messages)
                    if chain.history_manager
                    else 0
                )

                async for chunk in chain.astream({"input": user_input}):
                    print(chunk, end="", flush=True)
                print("\n")

                if chain.history_manager:
                    new_msgs = chain.history_manager.raw_messages[history_start_index:]
                    for msg in new_msgs:
                        if isinstance(msg, ModelResponse):
                            for part in msg.parts:
                                if isinstance(part, ToolCallPart):
                                    logger.info(
                                        f" > [DECISION] Calling Tool: {part.tool_name}"
                                    )
                        elif isinstance(msg, ModelRequest):
                            for part in msg.parts:
                                if isinstance(part, ToolReturnPart):
                                    content_len = len(str(part.content))
                                    logger.info(
                                        f" > [RESULT] Tool '{part.tool_name}' returned {content_len} chars"
                                    )

            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Error during interaction: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize ChainLite: {e}")


async def main():
    parser = argparse.ArgumentParser(description="ChainLite Integration Test CLI")
    parser.add_argument(
        "--config", type=str, default="tests/prompts/integration_truncation_config.yaml"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found. Some tests might fail.")

    await integration_history_cli(args.config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
