import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

# Fallback if load_dotenv doesn't work or file not found, try to read manually
if not os.getenv("GOOGLE_API_KEY"):
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GOOGLE_API_KEY="):
                    os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip()
                    break
    except Exception:
        pass

agent = Agent("google-gla:gemini-3-flash-preview", system_prompt="Answer with 'Hello'.")


def test_sync_stream():
    print("Testing sync stream...")
    try:
        # run_stream_sync returns a context manager in recent versions?
        # Or just the result? The code in core.py uses it as a return value, not context manager.
        # "result = agent.run_stream_sync(...)"
        # "for chunk in result.stream_text(delta=True):"
        # So it returns a result object.
        with agent.run_stream_sync("Say Hi") as result:
            print(f"Result type: {type(result)}")
            print(f"Dir result: {dir(result)}")

            if hasattr(result, "stream_responses"):
                print("Has stream_responses")
                for msg in result.stream_responses():
                    print(f"Message type: {type(msg)}")
                    if hasattr(msg, "parts"):
                        print(f"Message parts: {len(msg.parts)}")
            else:
                print("No stream_responses")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_sync_stream()
