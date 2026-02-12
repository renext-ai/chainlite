# ChainLite

ChainLite is a lightweight, configuration-driven utility for building LLM-powered applications. It wraps **Pydantic AI** to provide a simplified interface for managing LLM interactions, conversation history, and structured outputs.

## Key Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google Vertex/Gemini, Mistral, and Ollama.
- **YAML Configuration**: Define your agent's behavior, model parameters, and prompts in simple YAML files.
- **Advanced Templating**: Full support for **Jinja2** templating in prompts, enabling complex logic (conditionals, loops) directly in your configuration.
- **Conversation History**: Built-in support for persistent conversation history using Redis, with automatic session management.
- **Streaming Support**: Real-time response streaming with `astream`.
- **Structured Outputs**: Easy integration with Pydantic models for structured data extraction.
- **Async First**: Built on top of `asyncio` for high-performance applications, featuring non-blocking I/O for media processing.

## Installation

Since ChainLite is not currently published on PyPI, you can install it directly from the source.

### Prerequisites

- Python 3.10+
- (Optional) Redis server for persistent history

### Install from Source

**Option 1: Clone and Install**

```bash
git clone https://github.com/aidenpearce/chainlite.git
cd chainlite
pip install -e .
```

**Option 2: Direct Install from GitHub**

```bash
# Install latest (main branch)
pip install "git+https://github.com/aidenpearce/chainlite.git"

# Install specific version (Note: tags are prefixed with 'chainlite-v')
pip install "git+https://github.com/aidenpearce/chainlite.git@chainlite-v0.3.0"
pixi add --pypi "chainlite @ git+https://github.com/aidenpearce/chainlite.git@chainlite-v0.3.0"
```

## Quick Start

### Basic Usage with YAML

Create a config file `agent.yaml`:

```yaml
config_name: "my_assistant"
llm_model_name: "openai:gpt-3.5-turbo" # or "anthropic:claude-3-sonnet-20240229", "ollama:llama3"
system_prompt: "You are a helpful assistant."
prompt: "{{ input }}"
temperature: 0.7
use_history: true
session_id: "my_session"
# redis_url: "redis://localhost:6379/0"  # Uncomment to use Redis
```

Run it in Python:

> **Note for Jupyter Notebook / Interactive Users:**
> If you are running this in a Jupyter Notebook, IPython, or Streamlit, the event loop is already running. Do not use `asyncio.run()`. Instead, use `await agent.arun(...)` directly.

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

### Structured Output with YAML

You can define a schema for structured output directly in your YAML configuration using `output_parser`. This allows you to extract specific information from the model's response.

Create a config file `structured_agent.yaml`:

```yaml
config_name: "extractor"
llm_model_name: "openai:gpt-3.5-turbo"
system_prompt: "You are a helpful assistant that extracts information."
prompt: "Extract the name and age from this text: {{ text }}"
output_parser:
  - name: "The name of the person."
  - age:
      - type: "int"
        description: "The age of the person."
```

Run it in Python:

```python
async def main():
    agent = ChainLite.load_config_from_yaml("structured_agent.yaml")

    # The response will be a dictionary matching the schema
    response = await agent.arun({"text": "John Doe is 30 years old."})
    print(response)
    # Output: {'name': 'John Doe', 'age': 30}
```

### Python Class Usage

You can also initialize `ChainLite` directly with a configuration object:

```python
from chainlite import ChainLite, ChainLiteConfig

config = ChainLiteConfig(
    llm_model_name="openai:gpt-4",
    system_prompt="You are a mathematics tutor.",
    prompt="Solve this problem: {{ problem }}",
    temperature=0.2
)

chain = ChainLite(config=config)
result = chain.run({"problem": "What is the square root of 144?"})
print(result)
```

## Tools & Dependencies

ChainLite integrates seamlessly with **Pydantic AI**'s tool system and dependency injection. You can define Python functions as tools and access runtime dependencies (like User Context, Database sessions) directly within them.

For detailed usage and examples of how to bind tools and inject dependencies, please refer to [docs/tools.md](docs/tools.md).

## Configuration Reference

The `ChainLiteConfig` support the following key parameters:

| Parameter        | Type                   | Description                                                                                |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------------------ |
| `llm_model_name` | `str`                  | The model identifier (e.g., `openai:gpt-4`, `ollama:llama3`).                              |
| `system_prompt`  | `str`                  | The system instruction for the LLM.                                                        |
| `prompt`         | `str`                  | The user prompt template. Supports Jinja2 syntax (e.g., `{{ variable }}`, `{% if ... %}`). |
| `use_history`    | `bool`                 | Enable conversation history tracking.                                                      |
| `session_id`     | `str`                  | Unique ID for the conversation session (required if `use_history=True`).                   |
| `redis_url`      | `str`                  | URL for Redis instance to persist history.                                                 |
| `temperature`    | `float`                | Sampling temperature (0.0 to 1.0).                                                         |
| `model_settings` | `Dict[str, Any]`       | Provider-specific model settings passed through to Pydantic AI.                            |
| `max_retries`    | `int`                  | Number of retries for failed requests.                                                     |
| `output_parser`  | `List[Dict[str, Any]]` | Schema definition for structured output extraction.                                        |

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
   pip install -e .
   # or include test tools
   pip install -e ".[dev]"
   ```

2. **Run Tests**:
   ```bash
   pytest
   ```
   `integration` tests are excluded by default. To run integration tests:
   ```bash
   OPENAI_API_KEY=... pytest -m integration
   ```

## CI and Releases

- Pull requests run unit tests automatically via GitHub Actions.
- Integration tests are intentionally not part of default CI because they require provider credentials.
- PR titles are validated with semantic types (e.g. `feat`, `fix`, `docs`).
- Release Please automatically prepares version bumps and changelog updates via release PRs.
- A publish workflow releases package artifacts to PyPI when a GitHub Release is published.
- PyPI publish is gated by repository variable `PYPI_PUBLISH_ENABLED=true`; otherwise workflow logs a skip and exits successfully.

### Tagging Convention

Due to the use of `release-please` in manifest mode, git tags are prefixed with the package name:

- Format: `chainlite-vX.Y.Z` (e.g., `chainlite-v0.3.0`)
- **Important**: When referencing tags in `pip install` or CI/CD scripts, ensure you use the full `chainlite-v` prefix.

### Release Policy

- `fix` changes produce `PATCH` releases.
- `feat` changes produce `MINOR` releases.
- Breaking changes (`!`, for example `feat!:`) produce `MAJOR` releases.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, test strategy, and PR expectations.

## License

MIT License
