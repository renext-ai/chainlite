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
