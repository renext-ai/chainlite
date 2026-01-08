import asyncio
from chainlite import ChainLite
from loguru import logger
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


async def test_basic_yaml():
    logger.info("--- Testing Basic YAML Config ---")
    chain = ChainLite.load_config_from_yaml("tests/test_basic.yaml")
    response = await chain.arun({"topic": "artificial intelligence"})
    logger.info(f"Basic Response: {response}")
    logger.info("---------------------------------")


async def test_structured_yaml():
    logger.info("--- Testing Structured YAML Config ---")
    chain = ChainLite.load_config_from_yaml("tests/test_structured.yaml")
    response = await chain.arun({"text": "John Doe is 30 years old."})
    logger.info(f"Structured Response: {response}")
    if (
        isinstance(response, dict)
        and response.get("name") == "John Doe"
        and response.get("age") == 30
    ):
        logger.info("Structured YAML test passed!")
    else:
        logger.error("Structured YAML test failed!")
    logger.info("--------------------------------------")


async def test_streaming_yaml():
    logger.info("--- Testing Streaming from YAML Config ---")
    chain = ChainLite.load_config_from_yaml("tests/test_basic.yaml")
    logger.info("Streaming Response:")
    async for chunk in chain.astream({"topic": "ocean"}):
        print(chunk, end="", flush=True)
    print("\n-------------------------------------------")


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
