import asyncio
from chainlite import ChainLite, ChainLiteConfig
from dotenv import load_dotenv

load_dotenv()


async def main():
    print("Initializing ChainLite...")
    config = ChainLiteConfig(llm_model_name="openai:gpt-4o-mini", use_history=False)
    chain = ChainLite(config)
    print("Running chain.arun...")
    try:
        result = await chain.arun({"input": "Hello, simply reply 'Hi'."})
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
