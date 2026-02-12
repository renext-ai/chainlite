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


async def interactive_chat(config_path: str):
    """
    Main loop for interactive chat with a ChainLite orchestrator using config-driven sub-agents.
    No manual @agent.tool registration needed — sub-agents are declared in YAML.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    try:
        # Load chain — sub-agents are auto-registered as tools from YAML
        chain = ChainLite.load_config_from_yaml(config_path)

        # Log registered sub-agent tools
        if chain.config.sub_agents:
            tool_names = [sa.name for sa in chain.config.sub_agents]
            logger.info(f"Sub-agent tools registered: {', '.join(tool_names)}")

        # Enable history for multi-turn chat
        if not chain.config.use_history:
            logger.warning(
                "History was not enabled in config. Enabling it now for multi-turn chat."
            )
            chain.config.use_history = True
            from chainlite.core import HistoryManager

            chain.history_manager = HistoryManager(
                session_id=chain.config.session_id or "interactive_orchestrator",
                redis_url=chain.config.redis_url,
            )

        logger.info("Type 'exit', 'quit', or 'clear' to manage the session.")
        print("\n" + "=" * 50)
        print("  ChainLite Orchestrator (Config-Driven Sub-Agents)")
        print("=" * 50 + "\n")

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

                # Stream the response
                async for chunk in chain.astream({"input": user_input}):
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
                                        f" > [DECISION] Calling sub-agent: {part.tool_name} with args: {part.args}"
                                    )
                        elif isinstance(msg, ModelRequest):
                            for part in msg.parts:
                                if isinstance(part, ToolReturnPart):
                                    logger.info(
                                        f" > [RESULT] Sub-agent '{part.tool_name}' returned: {part.content}"
                                    )

            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Streaming error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize ChainLite: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="ChainLite Interactive Orchestrator (Config-Driven Sub-Agents)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tests/prompts/orchestrator.yaml",
        help="Path to the orchestrator YAML config file",
    )
    args = parser.parse_args()

    await interactive_chat(args.config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
