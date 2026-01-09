import asyncio
import os
from loguru import logger
from chainlite import ChainLite, ChainLiteConfig


# This script is designed for functional testing of the ChainLite core methods.
# It will make real API calls to the specified model provider.
#
# PRE-REQUISITES:
# 1. Ensure you have the necessary libraries installed:
#    pip install chainlite pydantic-ai python-dotenv
# 2. Create a .env file by copying .env.example and adding your API key.
#    The script will automatically load it.
# 3. Alternatively, you can export the environment variable directly:
#    export OPENAI_API_KEY='your_api_key_here'

# --- Configuration ---
# You can change this configuration to test different models or providers.
# Make sure the llm_model_name is prefixed with the correct provider.
test_config = ChainLiteConfig(
    llm_model_name="openai:gpt-3.5-turbo",
    prompt="Tell me a short, one-sentence joke about a computer.",
    model_settings={"temperature": 0.2},
    config_name="functional_test",
)

import pytest


@pytest.fixture
def chain():
    return ChainLite(config=test_config)


# --- Test Functions ---


def test_run(chain: ChainLite):
    """Tests the synchronous .run() method."""
    logger.info("--- Testing .run() ---")
    try:
        response = chain.run({})
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    logger.info("-" * 22 + "\n")


async def test_arun(chain: ChainLite):
    """Tests the asynchronous .arun() method."""
    logger.info("--- Testing .arun() ---")
    try:
        response = await chain.arun({})
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    logger.info("-" * 23 + "\n")


def test_stream(chain: ChainLite):
    """Tests the synchronous .stream() method."""
    logger.info("--- Testing .stream() ---")
    try:
        print("Response: ", end="", flush=True)
        for chunk in chain.stream({}):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 25 + "\n")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print("-" * 25 + "\n")


async def test_astream(chain: ChainLite):
    """Tests the asynchronous .astream() method."""
    logger.info("--- Testing .astream() ---")
    try:
        print("Response: ", end="", flush=True)
        async for chunk in chain.astream({}):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 26 + "\n")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print("-" * 26 + "\n")


async def test_structured_output():
    """Tests structured output with output_parser configuration."""
    logger.info("--- Testing Structured Output ---")
    try:
        structured_config = ChainLiteConfig(
            llm_model_name="openai:gpt-3.5-turbo",
            prompt="Tell me a joke about {{ topic }}",
            temperature=0.7,
            output_parser=[
                {"setup": "The setup of the joke"},
                {"punchline": "The punchline of the joke"},
            ],
            config_name="structured_test",
        )
        chain = ChainLite(config=structured_config)
        response = await chain.arun({"topic": "programming"})
        logger.info(f"Structured Response: {response}")
        assert isinstance(response, dict), "Response should be a dictionary"
        assert "setup" in response, "Response should have 'setup' key"
        assert "punchline" in response, "Response should have 'punchline' key"
        logger.info("Structured output test passed!")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    logger.info("-" * 35 + "\n")


async def main():
    """Main function to run all tests."""
    # Load environment variables from .env file if python-dotenv is installed
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # Rely on manually set environment variables

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set.")
        logger.error("Please set the variable and try again.")
        return

    # Initialize ChainLite
    try:
        chain = ChainLite(config=test_config)
    except Exception as e:
        logger.error(f"Failed to initialize ChainLite: {e}")
        return

    # Run the tests
    test_run(chain)
    test_stream(chain)
    await test_arun(chain)
    await test_astream(chain)
    await test_structured_output()


if __name__ == "__main__":
    logger.info("Starting ChainLite Functional Test (Pydantic AI)...\n")
    asyncio.run(main())
