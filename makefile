chat:
	pixi run python tests/interactive_chat.py --config tests/chat_agent.yaml
chat-image:
	pixi run python tests/interactive_chat_with_image.py --config tests/chat_agent.yaml
manual-image:
	pixi run python tests/manual_image_test.py
