import asyncio
import os
import argparse
from dataclasses import dataclass
from loguru import logger
from dotenv import load_dotenv
from chainlite import ChainLite
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelResponse,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)

# Load environment variables
load_dotenv()


@dataclass
class UserContext:
    username: str
    user_id: int
    location: str


async def interactive_chat(config_path: str):
    """
    Main loop for interactive chat with a ChainLite agent with Tools and Deps.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    # Initialize ChainLite from YAML
    try:
        # Load chain
        chain = ChainLite.load_config_from_yaml(config_path)

        # ---------------------------------------------------------
        # Dynamically Setup Tools and Deps
        # ---------------------------------------------------------

        # Note: If the agent was initialized without a deps_type,
        # passing deps might cause runtime warnings or errors depending on pydantic-ai version,
        # unless we explicitly update the agent or if it defaults to Any.

        # Register tools
        @chain.agent.tool
        def get_user_info(ctx: RunContext[UserContext]) -> str:
            """Get information about the current user (deps)."""
            return f"User: {ctx.deps.username} (ID: {ctx.deps.user_id}) from {ctx.deps.location}"

        @chain.agent.tool
        def get_weather(ctx: RunContext[UserContext], city: str) -> str:
            """Get the weather for a specific city. If city is not provided, use user's location from deps."""
            target_city = city or ctx.deps.location
            # Mock weather logic
            return f"The weather in {target_city} is Sunny, 25°C."

        logger.info(f"Initialized agent with tools: {chain.config.config_name}")
        logger.info("Tools registered: get_user_info, get_weather")

        # ---------------------------------------------------------

        # Check/Enable History
        if not chain.config.use_history:
            logger.warning(
                "History was not enabled in config. Enabling it now for multi-turn chat."
            )
            chain.config.use_history = True
            from chainlite.core import HistoryManager

            chain.history_manager = HistoryManager(
                session_id=chain.config.session_id or "interactive_session_tools",
                redis_url=chain.config.redis_url,
            )

        logger.info("Type 'exit', 'quit', or 'clear' to manage the session.")
        print("\n" + "=" * 50)
        print("  ChainLite Interactive Chat (Tools + Deps)")
        print("=" * 50 + "\n")

        # Create Dependencies
        current_deps = UserContext(username="Aiden", user_id=42, location="Taiwan")
        logger.info(f"Loaded Deps: {current_deps}")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                if chain.history_manager:
                    chain.history_manager.clear()
                    print("\n[Chat history cleared]\n")
                continue

            if not user_input:
                continue

            print("\nAgent: ", end="", flush=True)
            try:
                # Snapshot history length
                history_start_index = (
                    len(chain.chat_history_messages)
                    if chain.chat_history_messages
                    else 0
                )

                # Use astream for real-time output, passing deps
                async for chunk in chain.astream(
                    {"input": user_input}, deps=current_deps
                ):
                    print(chunk, end="", flush=True)
                print("\n")

                # Inspect and print tool calls from the new history
                if chain.chat_history_messages:
                    new_msgs = chain.chat_history_messages[history_start_index:]
                    for msg in new_msgs:
                        if isinstance(msg, ModelResponse):
                            for part in msg.parts:
                                if isinstance(part, ToolCallPart):
                                    logger.info(
                                        f" > [DECISION] Calling Tool: {part.tool_name} with args: {part.args}"
                                    )
                        elif isinstance(msg, ModelRequest):
                            for part in msg.parts:
                                if isinstance(part, ToolReturnPart):
                                    logger.info(
                                        f" > [RESULT] Tool '{part.tool_name}' returned: {part.content}"
                                    )

            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Streaming error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize ChainLite: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="ChainLite Interactive Chat Test Tool (Tools+Deps)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tests/chat_agent.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment or .env file.")

        # Just a warning as user might have it in some other way, but usually required.
        # But let's return if really missing to avoid crashing inside.
        # However, .env is loaded.

    await interactive_chat(args.config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
