chat:
	pixi run python tests/interactive_chat.py --config tests/chat_agent.yaml
chat-image:
	pixi run python tests/interactive_chat_with_image.py --config tests/chat_agent.yaml
chat-tools-deps:
	pixi run python tests/interactive_chat_tools_deps.py --config tests/chat_agent.yaml
manual-image:
	pixi run python tests/manual_image_test.py
test:
	pytest
test-integration:
	OPENAI_API_KEY=$$OPENAI_API_KEY pytest -m integration
