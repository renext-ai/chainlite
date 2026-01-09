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
