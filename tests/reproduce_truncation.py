from chainlite.history import HistoryManager
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from datetime import datetime, timezone


def test_smart_truncation():
    print("Initializing HistoryManager with max_messages=3")
    hm = HistoryManager(session_id="test_sess", max_messages=3)

    msgs = []

    # 0. User
    msgs.append(ModelRequest(parts=[UserPromptPart(content="0. Hi")]))

    # 1. Model Tool Call
    msgs.append(
        ModelResponse(
            parts=[ToolCallPart(tool_name="foo", args="bar", tool_call_id="1")]
        )
    )

    # 2. User Tool Return
    msgs.append(
        ModelRequest(
            parts=[ToolReturnPart(tool_name="foo", content="baz", tool_call_id="1")]
        )
    )

    # 3. Model Response
    msgs.append(ModelResponse(parts=[TextPart(content="3. Result")]))

    # 4. User Follow up
    msgs.append(ModelRequest(parts=[UserPromptPart(content="4. Follow up")]))

    print(f"Adding {len(msgs)} messages...")
    hm.add_messages(msgs)

    print(f"History length: {len(hm.messages)}")
    for i, msg in enumerate(hm.messages):
        print(f"Msg {i}: {msg}")

    # Assertions
    if not hm.messages:
        print("FAIL: History is empty")
        return

    first_msg = hm.messages[0]
    if isinstance(first_msg, ModelRequest):
        print("First message is ModelRequest.")
        # Check if it is a User prompt
        if isinstance(first_msg.parts[0], UserPromptPart):
            print("PASS: First message is UserPrompt.")
        else:
            print(f"FAIL: First message part is {type(first_msg.parts[0])}")
    else:
        print(f"FAIL: First message is {type(first_msg)}")

    expected_len = 1  # based on logic above
    if len(hm.messages) == expected_len:
        print(
            f"PASS: Length is {expected_len} as expected (aggressive truncation to find safe user start)."
        )
    else:
        print(
            f"WARN: Length is {len(hm.messages)}, expected {expected_len}. Check if logic allows ModelResponse start?"
        )


if __name__ == "__main__":
    test_smart_truncation()
