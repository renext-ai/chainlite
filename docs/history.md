# ChainLite History Management

ChainLite provides a robust system for managing conversation history, ensuring that long-running sessions do not exceed the context window limits of LLMs. This system includes modular truncation strategies, automatic summarization, and history export capabilities for debugging.

## Configuration

History management is configured via the `history_truncator_config` section in your YAML configuration file.

```yaml
# Example Configuration
history_truncator_config:
  post_run_compaction:
    mode: "simple" # Options: "simple", "auto", "custom"
    truncation_threshold: 5000
    start_run: 1   # Apply post-run compaction from Nth run
  # Optional in-run compaction
  # in_run_compaction:
  #   mode: "simple"
  #   truncation_threshold: 3000
  #   start_iter: 2
  #   start_run: 1
  #   max_concurrency: 4
```

### Modes

ChainLite supports three modes for handling long history:

#### 1. Simple Truncation (`post_run_compaction.mode: "simple"`)

This mode truncates tool outputs that exceed the `truncation_threshold` by keeping the first `N` characters and appending a suffix indicating the number of omitted characters.

**Pros:** Fast, zero cost.
**Cons:** Loss of information; context might be lost.

#### 2. Auto Summarization (`post_run_compaction.mode: "auto"`)

This mode uses the **same LLM model** as the primary agent to summarize tool outputs that exceed the threshold.

**Mechanism:**

- When a tool output is too long, the system pauses.
- It sends the user's question and the tool's raw output to the LLM with a summarization prompt.
- The LLM returns a concise summary.
- The original long output is replaced in the conversation history with `[Summarized Output]: <summary>`.
- The **Raw History** (used for full exports) retains the original complete output.

**Pros:** Preserves semantic meaning; automatic.
**Cons:** Incurs additional token costs and latency.

#### 3. Custom Summarization (`post_run_compaction.mode: "custom"`)

This mode allows you to define a **separate ChainLite agent** dedicated to summarization. This is useful if you want to use a cheaper or faster model (e.g., `gpt-3.5-turbo`) for summarizing while using a powerful model (e.g., `gpt-4o`) for the main conversation.

**Configuration:**
You must provide either a path to a config file OR a config dictionary.

```yaml
history_truncator_config:
  post_run_compaction:
    mode: "custom"
    truncation_threshold: 5000
    summarizor_config_path: "tests/prompts/custom_summarizer.yaml"
  # OR
  # post_run_compaction:
  #   mode: "custom"
  #   summarizor_config_dict:
  #     llm_model_name: "openai:gpt-3.5-turbo"
  #     system_prompt: "Summarize this concisely."
```

## History Export

ChainLite allows you to export conversation history for review, debugging, or dataset creation.

### Usage (Python)

```python
# Export all history (Raw and Truncated) in Markdown format
paths = chain.history_manager.export(export_type="all", export_format="markdown")
print(f"Exported to: {paths}")
```

### Export Types

- **`full` (Raw)**: Contains the exact, unmodified output from all tools. Essential for debugging what actually happened.
- **`truncated`**: Contains the history exactly as the LLM sees it (with summaries). Useful for understanding why the LLM responded in a certain way.
- **`all`**: Exports both versions.

### Export Formats

- **`markdown`**: Human-readable format with role headers and code blocks.
- **`json`**: Machine-readable format compatible with Pydantic serialization.
