import os
import argparse
from loguru import logger
from dotenv import load_dotenv
from chainlite import ChainLite

# Load environment variables
load_dotenv()


def sync_chat(config_path: str):
    """
    Main loop for synchronous interactive chat with a ChainLite agent.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    # Initialize ChainLite from YAML
    try:
        # Load and ensure history is enabled for multi-turn
        chain = ChainLite.load_config_from_yaml(config_path)
        if not chain.config.use_history:
            logger.warning(
                "History was not enabled in config. Enabling it now for multi-turn chat."
            )
            chain.config.use_history = True
            # Re-initialize history manager if needed
            from chainlite.core import HistoryManager

            chain.history_manager = HistoryManager(
                session_id=chain.config.session_id or "sync_interactive_session",
                redis_url=chain.config.redis_url,
            )

        logger.info(f"Initialized agent: {chain.config.config_name}")
        logger.info("Type 'exit', 'quit', or 'clear' to manage the session.")
        print("\n" + "=" * 50)
        print("  ChainLite Synchronous Chat Test")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

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
                # Use stream (sync) for real-time output
                for chunk in chain.stream(
                    {
                        "input": user_input,
                        "secret_number": 666,
                    }
                ):
                    print(chunk, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Streaming error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize ChainLite: {e}")


def main():
    parser = argparse.ArgumentParser(description="ChainLite Sync Chat Test Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="tests/chat_agent.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    # Check for API key
    if (
        not os.getenv("OPENAI_API_KEY")
        and not os.getenv("GEMINI_API_KEY")
        and not os.getenv("GOOGLE_API_KEY")
    ):
        # Just a warning, provider might use different key
        logger.warning(
            "No standard API KEY found in environment. Ensure your provider is configured."
        )

    sync_chat(args.config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
