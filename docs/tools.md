## Advanced Usage: Adding Custom Tools

While ChainLite manages prompts via YAML, you can easily attach custom Python functions (tools) to the agent using the underlying Pydantic AI decorator. This is essential for agents that need to perform actions or fetch data.

### Example: Weather Bot with Tools

**1. Define `agent.yaml`:**
```yaml
config_name: "weather_bot"
llm_model_name: "openai:gpt-4o"
system_prompt: "You are a helpful assistant. Use the available tools to answer questions."
prompt: "{{ input }}"
```


**2. Define Tools in Python:**
```python
from chainlite import ChainLite
from pydantic_ai import RunContext

# Load configuration
chain = ChainLite.load_config_from_yaml("agent.yaml")

# Register a tool directly to the underlying agent
@chain.agent.tool
def get_weather(ctx: RunContext, location: str) -> str:
    """Get the weather for a specific location."""
    # Your logic here (e.g., API call)
    return f"The weather in {location} is sunny."

# Run the agent
async def main():
    response = await chain.arun({"input": "What's the weather in Taipei?"})
    print(response)
```

### Injecting Dependencies (Deps)

ChainLite fully supports `pydantic-ai`'s dependency injection system, allowing you to pass runtime context (like user info, database connections, or request metadata) into your tools safely and transparently.

**1. Define your Dependencies:**
You can use any Python type, but `dataclasses` or Pydantic models are recommended.

```python
from dataclasses import dataclass

@dataclass
class UserContext:
    username: str
    user_id: int
    location: str
```

**2. Register Tools that use Deps:**
Annotate the `ctx` argument with `RunContext[YourDepType]`.

```python
from pydantic_ai import RunContext

@chain.agent.tool
def get_user_info(ctx: RunContext[UserContext]) -> str:
    """Get information about the current user from deps."""
    return f"User: {ctx.deps.username} (ID: {ctx.deps.user_id})"

@chain.agent.tool
def get_local_weather(ctx: RunContext[UserContext]) -> str:
    """Get weather for the user's location."""
    # Access runtime dependency
    loc = ctx.deps.location
    return f"Weather in {loc} is Rainy."
```

**3. Pass Deps at Runtime:**
Pass the `deps` argument when calling `run`, `arun`, `stream`, or `astream`.

```python
current_user = UserContext(username="Alice", user_id=123, location="London")

# The agent can now use tools that access 'current_user'
response = await chain.arun(
    {"input": "Who am I and what is the weather here?"},
    deps=current_user
)
print(response)
```

## Config-Driven Sub-Agents

Instead of manually registering tools with `@chain.agent.tool`, you can declare sub-agents directly in YAML. Each sub-agent is another ChainLite config that gets automatically registered as a tool on the parent agent during `setup_chain()`.

### YAML Syntax

**`orchestrator.yaml`:**
```yaml
config_name: "orchestrator"
llm_model_name: "openai:gpt-4"
system_prompt: "You are a coordinator. Use the available tools to help the user."
prompt: "{{ input }}"
sub_agents:
  - name: "translate"
    config: "translator.yaml"
    description: "Translate text to French."
  - name: "summarize"
    config: "summarizer.yaml"
    description: "Summarize long text into one sentence."
```

**`translator.yaml`:**
```yaml
llm_model_name: "openai:gpt-3.5-turbo"
system_prompt: "You are a translator. Translate the input text to French."
prompt: "{{ input }}"
```

**`summarizer.yaml`:**
```yaml
llm_model_name: "openai:gpt-3.5-turbo"
system_prompt: "You are a summarizer. Provide a one-sentence summary."
prompt: "{{ input }}"
```

### Usage

```python
from chainlite import ChainLite

chain = ChainLite.load_config_from_yaml("orchestrator.yaml")

# That's it! Sub-agents are already registered as tools.
# The orchestrator LLM can now call "translate" and "summarize" as needed.
response = await chain.arun({"input": "Translate 'hello world' to French."})
```

### How It Works

- Each sub-agent entry declares a `name` (tool name), `config` (YAML path), and `description` (shown to the LLM).
- Config paths are resolved **relative to the parent YAML file** when using `load_config_from_yaml()`.
- Sub-agents are **lazily instantiated** on first tool call, not at load time.
- Each tool takes a single `input: str` parameter and passes it as `{"input": input}` to the sub-agent.
- Sub-agent calls are **stateless** — each invocation is independent (no conversation history).
- You can **combine** config-driven sub-agents with manual `@chain.agent.tool` decorators on the same chain.
