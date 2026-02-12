import asyncio
import sys
import json
from pydantic_ai import Agent, RunContext


async def main():
    try:
        agent = Agent("openai:gpt-4o-mini", instructions="You are a helpful assistant.")

        @agent.tool
        def get_weather(ctx: RunContext[None], city: str) -> str:
            """Get the weather for a city."""
            return f"Sunny in {city}"

        if hasattr(agent, "_function_toolset"):
            toolset = agent._function_toolset
            for name, tool in toolset.tools.items():
                print(f"\nTool: {name}")
                if hasattr(tool, "function_schema"):
                    # Pydantic AI's FunctionSchema should have model_dump()
                    try:
                        schema_dict = tool.function_schema.model_dump()
                        print(f"  function_schema: {json.dumps(schema_dict, indent=2)}")
                    except AttributeError:
                        # Fallback for older Pydantic
                        schema_dict = tool.function_schema.dict()
                        print(f"  function_schema: {json.dumps(schema_dict, indent=2)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
