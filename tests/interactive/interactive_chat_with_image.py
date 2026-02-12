import asyncio
import os
import argparse
from loguru import logger
from dotenv import load_dotenv
from chainlite import ChainLite

# Load environment variables
load_dotenv()

async def interactive_chat(config_path: str):
    """
    Main loop for interactive chat with a ChainLite agent that supports Image Inputs.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    # Initialize ChainLite from YAML
    try:
        # Load and ensure history is enabled for multi-turn
        chain = ChainLite.load_config_from_yaml(config_path)
        if not chain.config.use_history:
            logger.warning("History was not enabled in config. Enabling it now for multi-turn chat.")
            chain.config.use_history = True
            # Re-initialize history manager if needed
            from chainlite.core import HistoryManager
            chain.history_manager = HistoryManager(
                session_id=chain.config.session_id or "interactive_session",
                redis_url=chain.config.redis_url
            )
        
        logger.info(f"Initialized agent: {chain.config.config_name}")
        logger.info("Type 'exit', 'quit', or 'clear' to manage the session.")
        print("\n" + "="*50)
        print("  ChainLite Interactive Chat w/ Image Support")
        print("="*50 + "\n")

        while True:
            # 1. Get Text Input
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

            # 2. Get Optional Images
            images_input = input("Images (URL/local/base64, comma-sep): ").strip()
            
            input_data = {"input": user_input}
            if images_input:
                # Split by comma and strip whitespace
                images_list = [img.strip() for img in images_input.split(",") if img.strip()]
                input_data["images"] = images_list
                

            print("\nAgent: ", end="", flush=True)
            try:
                # Use astream for real-time output
                async for chunk in chain.astream(input_data):
                    print(chunk, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Streaming error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize ChainLite: {e}")

async def main():
    parser = argparse.ArgumentParser(description="ChainLite Interactive Chat Test Tool (Multimodal)")
    parser.add_argument(
        "--config", 
        type=str, 
        default="tests/chat_agent.yaml",
        help="Path to the YAML config file"
    )
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment or .env file.")
        return

    await interactive_chat(args.config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
