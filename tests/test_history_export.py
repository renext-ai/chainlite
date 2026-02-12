import asyncio
import os
from chainlite.history import HistoryManager
from chainlite.trunctors import SimpleTrunctor
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart


async def test_history_export():
    print("\n--- Testing History Export ---")
    hm = HistoryManager(session_id="test_export", max_messages=10)

    # Add some messages
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="Hello world")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="test_tool",
                    content="This is a very long tool output " * 10,
                    tool_call_id="call_x",
                )
            ]
        ),
    ]
    hm.add_messages(msgs)

    # Export JSON
    json_files = hm.export(
        export_type="all", export_format="json", output_dir="temp_export"
    )
    print(f"Exported JSON files: {json_files}")

    # Export Markdown
    md_files = hm.export(
        export_type="all", export_format="markdown", output_dir="temp_export"
    )
    print(f"Exported Markdown files: {md_files}")

    for f in json_files + md_files:
        assert os.path.exists(f)
        print(f"Verified existence: {f}")
        # Clean up
        os.remove(f)

    if os.path.exists("temp_export"):
        os.rmdir("temp_export")

    print("PASS: History export verified.")


if __name__ == "__main__":
    if not os.path.exists("temp_export"):
        os.makedirs("temp_export")
    asyncio.run(test_history_export())
