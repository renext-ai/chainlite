# ChainLite
ChainLite is a lightweight, configuration-driven utility for building LLM-powered applications. It wraps **Pydantic AI** to provide a simplified interface for managing LLM interactions, conversation history, and structured outputs.

## Key Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google Vertex/Gemini, Mistral, and Ollama.
- **YAML Configuration**: Define your agent's behavior, model parameters, and prompts in simple YAML files.
- **Conversation History**: Built-in support for persistent conversation history using Redis, with automatic session management.
- **Streaming Support**: Real-time response streaming with `astream`.
- **Structured Outputs**: Easy integration with Pydantic models for structured data extraction.
- **Async First**: Built on top of `asyncio` for high-performance applications.

## Installation

Since ChainLite is not currently published on PyPI, you can install it directly from the source.

### Prerequisites
- Python 3.8+
- (Optional) Redis server for persistent history

### Install from Source

**Option 1: Clone and Install**
```bash
git clone https://github.com/renext-ai/chainlite.git
cd chainlite
pip install -e .
```

**Option 2: Direct Install from GitHub**
```bash
pip install "git+https://github.com/renext-ai/chainlite.git"
pixi add --pypi "chainlite @ git+https://github.com/renext-ai/chainlite.git"
```

## Quick Start

### Basic Usage with YAML

Create a config file `agent.yaml`:

```yaml
config_name: "my_assistant"
llm_model_name: "openai:gpt-3.5-turbo"  # or "anthropic:claude-3-sonnet-20240229", "ollama:llama3"
system_prompt: "You are a helpful assistant."
prompt: "{input}"
temperature: 0.7
use_history: true
session_id: "my_session"
# redis_url: "redis://localhost:6379/0"  # Uncomment to use Redis
```

Run it in Python:

```python
import asyncio
import os
from dotenv import load_dotenv
from chainlite import ChainLite

# Load API keys (ensure OPENAI_API_KEY is set)
load_dotenv()

async def main():
    # Initialize from config
    agent = ChainLite.load_config_from_yaml("agent.yaml")
    
    # Run a simple query
    response = await agent.arun({"input": "Hello! Who are you?"})
    print(response)

    # Streaming example
    print("\nStreaming response:")
    async for chunk in agent.astream({"input": "Tell me a short story."}):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### Python Class Usage

You can also initialize `ChainLite` directly with a configuration object:

```python
from chainlite import ChainLite, ChainLiteConfig

config = ChainLiteConfig(
    llm_model_name="openai:gpt-4",
    system_prompt="You are a mathematics tutor.",
    prompt="Solve this problem: {problem}",
    temperature=0.2
)

chain = ChainLite(config=config)
result = chain.run_sync({"problem": "What is the square root of 144?"})
print(result)
```

## Configuration Reference

The `ChainLiteConfig` support the following key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm_model_name` | `str` | The model identifier (e.g., `openai:gpt-4`, `ollama:llama3`). |
| `system_prompt` | `str` | The system instruction for the LLM. |
| `prompt` | `str` | The user prompt template. Supports `{variable}` interpolation. |
| `use_history` | `bool` | Enable conversation history tracking. |
| `session_id` | `str` | Unique ID for the conversation session (required if `use_history=True`). |
| `redis_url` | `str` | URL for Redis instance to persist history. |
| `temperature` | `float` | Sampling temperature (0.0 to 1.0). |
| `max_tokens` | `int` | Maximum tokens to generate. |
| `retries` | `int` | Number of retries for failed requests. |

## Development

### Using Pixi (Recommended)

This project uses [Pixi](https://prefix.dev/docs/pixi/overview) for dependency management and environment handling.

1. **Install Pixi**:
   Follow the [official installation guide](https://prefix.dev/docs/pixi/installation).

2. **Install Dependencies**:
   ```bash
   pixi install
   ```

3. **Run Scripts**:
   You can run python scripts directly using `pixi run`:
   ```bash
   pixi run python tests/interactive_chat.py --config tests/chat_agent.yaml
   ```
   
   Or spawn a shell with the environment activated:
   ```bash
   pixi shell
   ```

4. **Manage Dependencies**:
   ```bash
   # Add a new dependency
   pixi add <package_name>
   
   # Add a PyPI dependency
   pixi add --pypi <package_name>
   ```

### Using Pip (Alternative)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR
   pip install -e .
   ```

2. **Run Tests**:
   Ensure you have a `.env` file with necessary API keys (e.g., `OPENAI_API_KEY`).
   ```bash
   python tests/interactive_chat.py --config tests/chat_agent.yaml
   ```

## License

MIT License
