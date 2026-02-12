chat:
	pixi run python tests/interactive/interactive_chat.py --config tests/prompts/chat_agent.yaml
chat-image:
	pixi run python tests/interactive/interactive_chat_with_image.py --config tests/prompts/chat_agent.yaml
chat-tools-deps:
	pixi run python tests/interactive/interactive_chat_tools_deps.py --config tests/prompts/chat_agent.yaml
manual-image:
	pixi run python tests/interactive/manual_image_test.py
history-cli:
	pixi run python tests/interactive/integration_history_cli.py
test:
	pytest
test-integration:
	OPENAI_API_KEY=$$OPENAI_API_KEY pixi run pytest -m integration
