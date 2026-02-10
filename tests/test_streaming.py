import asyncio
from chainlite import ChainLite
from chainlite.streaming import stream_sse
from loguru import logger
from dotenv import load_dotenv
import pytest
import os

# Load environment variables
load_dotenv()

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def check_env():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        pytest.skip("OPENAI_API_KEY is not configured; skipping integration tests.")


def test_sync_stream():
    logger.info("--- Testing Synchronous .stream() ---")
    chain = ChainLite.load_config_from_yaml("tests/test_basic.yaml")
    logger.info("Sync Streaming Response:")
    for chunk in chain.stream({"topic": "a lightbulb"}):
        print(chunk, end="", flush=True)
    print("\n--------------------------------------")


async def test_sse_stream():
    logger.info("--- Testing SSE .stream_sse() ---")
    chain = ChainLite.load_config_from_yaml("tests/test_basic.yaml")
    logger.info("SSE Streaming Response:")
    async for event in stream_sse(chain, {"topic": "innovation"}):
        print(event, end="", flush=True)
    print("\n---------------------------------------------------------")


async def main():
    if (
        not os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here"
    ):
        logger.error("Please set OPENAI_API_KEY in .env file before running tests.")
        return

    await test_sse_stream()


if __name__ == "__main__":
    test_sync_stream()
    asyncio.run(main())
