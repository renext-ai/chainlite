import asyncio
from chainlite import ChainLite
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


async def test_basic_yaml():
    logger.info("--- Testing Basic YAML Config ---")
    chain = ChainLite.load_config_from_yaml("tests/prompts/test_basic.yaml")
    response = await chain.arun({"topic": "artificial intelligence"})
    logger.info(f"Basic Response: {response}")
    assert response is not None, "Response should not be None"
    assert len(str(response)) > 0, "Response should not be empty"
    logger.info("---------------------------------")


async def test_structured_yaml():
    logger.info("--- Testing Structured YAML Config ---")
    chain = ChainLite.load_config_from_yaml("tests/prompts/test_structured.yaml")
    target_age = 31
    input_text = f"John Doe is {target_age} years old."
    response = await chain.arun({"text": input_text})

    logger.info(f"Structured Response: {response}")

    assert (
        response.get("name") == "John Doe"
    ), f"Expected 'John Doe', got '{response.get('name')}'"

    assert (
        response.get("age") == target_age
    ), f"Expected age {target_age}, got {response.get('age')}"

    logger.info("Structured YAML test passed!")
    logger.info("--------------------------------------")


async def test_streaming_yaml():
    logger.info("--- Testing Streaming from YAML Config ---")
    chain = ChainLite.load_config_from_yaml("tests/prompts/test_basic.yaml")
    logger.info("Streaming Response:")
    collected_text = ""
    async for chunk in chain.astream({"topic": "ocean"}):
        print(chunk, end="", flush=True)
        collected_text += str(chunk)
    print("\n-------------------------------------------")

    assert len(collected_text) > 0, "Stream result should not be empty"


async def main():
    if (
        not os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here"
    ):
        logger.error("Please set OPENAI_API_KEY in .env file before running tests.")
        return

    await test_basic_yaml()
    await test_structured_yaml()
    await test_streaming_yaml()


if __name__ == "__main__":
    asyncio.run(main())
