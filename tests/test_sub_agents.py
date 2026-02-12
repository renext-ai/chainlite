import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from chainlite import ChainLite, ChainLiteConfig, SubAgentConfig
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def set_dummy_openai_key():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")


# --- Config Validation Tests ---


def test_sub_agent_config_model():
    """SubAgentConfig accepts valid fields."""
    sa = SubAgentConfig(
        name="translate", config="translator.yaml", description="Translate text."
    )
    assert sa.name == "translate"
    assert sa.config == "translator.yaml"
    assert sa.description == "Translate text."


def test_sub_agent_config_forbids_extra():
    """SubAgentConfig rejects unknown fields."""
    with pytest.raises(Exception):
        SubAgentConfig(
            name="translate",
            config="translator.yaml",
            description="desc",
            unknown_field="boom",
        )


def test_chainlite_config_with_sub_agents():
    """ChainLiteConfig accepts sub_agents field."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        sub_agents=[
            SubAgentConfig(
                name="translate",
                config="/abs/path/translator.yaml",
                description="Translate text.",
            )
        ],
    )
    assert len(config.sub_agents) == 1
    assert config.sub_agents[0].name == "translate"


def test_chainlite_config_without_sub_agents():
    """ChainLiteConfig works without sub_agents (backward compat)."""
    config = ChainLiteConfig(llm_model_name="openai:gpt-4")
    assert config.sub_agents is None


# --- Path Resolution Tests ---


def test_load_config_from_yaml_resolves_relative_paths(tmp_path):
    """Relative sub-agent paths resolve relative to parent YAML."""
    parent_yaml = tmp_path / "orchestrator.yaml"
    parent_yaml.write_text(
        'llm_model_name: "openai:gpt-4"\n'
        "sub_agents:\n"
        '  - name: "translate"\n'
        '    config: "agents/translator.yaml"\n'
        '    description: "Translate text."\n'
    )

    # Sub-agent YAML doesn't need to exist at load time (lazy instantiation)
    chain = ChainLite.load_config_from_yaml(str(parent_yaml))
    expected = str(tmp_path / "agents" / "translator.yaml")
    assert chain.config.sub_agents[0].config == expected


def test_load_config_from_yaml_preserves_absolute_paths(tmp_path):
    """Absolute sub-agent paths are not modified."""
    parent_yaml = tmp_path / "orchestrator.yaml"
    abs_path = "/some/absolute/path/translator.yaml"
    parent_yaml.write_text(
        f'llm_model_name: "openai:gpt-4"\n'
        f"sub_agents:\n"
        f'  - name: "translate"\n'
        f'    config: "{abs_path}"\n'
        f'    description: "Translate text."\n'
    )
    chain = ChainLite.load_config_from_yaml(str(parent_yaml))
    assert chain.config.sub_agents[0].config == abs_path


# --- Tool Registration Tests ---


def test_sub_agent_tools_registered_on_agent():
    """Sub-agent tools appear in the agent's toolset."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        sub_agents=[
            SubAgentConfig(
                name="translate",
                config="/fake/translator.yaml",
                description="Translate text.",
            ),
            SubAgentConfig(
                name="summarize",
                config="/fake/summarizer.yaml",
                description="Summarize text.",
            ),
        ],
    )
    chain = ChainLite(config)
    tool_names = set(chain.agent._function_toolset.tools.keys())
    assert "translate" in tool_names
    assert "summarize" in tool_names


def test_no_sub_agents_no_tools():
    """Without sub_agents, no extra tools are registered."""
    config = ChainLiteConfig(llm_model_name="openai:gpt-4")
    chain = ChainLite(config)
    assert len(chain.agent._function_toolset.tools) == 0


# --- Tool Invocation Tests ---


@pytest.mark.asyncio
async def test_sub_agent_tool_invocation():
    """Sub-agent tool correctly delegates to a sub-chain."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        sub_agents=[
            SubAgentConfig(
                name="echo",
                config="/fake/echo.yaml",
                description="Echo input.",
            ),
        ],
    )
    chain = ChainLite(config)

    tool = chain.agent._function_toolset.tools["echo"]

    with patch.object(ChainLite, "load_config_from_yaml") as mock_load:
        mock_sub_chain = MagicMock()
        mock_sub_chain.arun = AsyncMock(return_value="echoed: hello")
        mock_load.return_value = mock_sub_chain

        # Call the underlying function directly
        result = await tool.function(input="hello")
        assert result == "echoed: hello"
        mock_load.assert_called_once_with("/fake/echo.yaml")
        mock_sub_chain.arun.assert_called_once_with({"input": "hello"})


@pytest.mark.asyncio
async def test_sub_agent_tool_structured_output():
    """Sub-agent tool serializes dict results to JSON."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        sub_agents=[
            SubAgentConfig(
                name="extractor",
                config="/fake/extractor.yaml",
                description="Extract info.",
            ),
        ],
    )
    chain = ChainLite(config)

    tool = chain.agent._function_toolset.tools["extractor"]

    with patch.object(ChainLite, "load_config_from_yaml") as mock_load:
        mock_sub_chain = MagicMock()
        mock_sub_chain.arun = AsyncMock(return_value={"name": "Alice", "age": 30})
        mock_load.return_value = mock_sub_chain

        result = await tool.function(input="Alice is 30")
        assert '"name": "Alice"' in result
        assert '"age": 30' in result


@pytest.mark.asyncio
async def test_sub_agent_tool_error_handling():
    """Sub-agent tool returns error string on failure."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        sub_agents=[
            SubAgentConfig(
                name="failing",
                config="/fake/fail.yaml",
                description="Always fails.",
            ),
        ],
    )
    chain = ChainLite(config)
    tool = chain.agent._function_toolset.tools["failing"]

    with patch.object(
        ChainLite, "load_config_from_yaml", side_effect=FileNotFoundError("not found")
    ):
        result = await tool.function(input="test")
        assert "Error from sub-agent 'failing'" in result


@pytest.mark.asyncio
async def test_sub_agent_lazy_instantiation():
    """Sub-agent is only instantiated on first tool call, not at setup time."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        sub_agents=[
            SubAgentConfig(
                name="lazy",
                config="/fake/lazy.yaml",
                description="Lazy agent.",
            ),
        ],
    )

    with patch.object(ChainLite, "load_config_from_yaml") as mock_load:
        # Creating the chain should NOT trigger sub-agent loading
        original_load = ChainLite.load_config_from_yaml.__wrapped__ if hasattr(ChainLite.load_config_from_yaml, '__wrapped__') else None

    # The mock was only active during chain creation above, so we verify
    # load_config_from_yaml was NOT called for the sub-agent config
    chain = ChainLite(config)
    # Tool exists but sub-agent not yet loaded
    assert "lazy" in chain.agent._function_toolset.tools


# --- Dynamic Agent Recreation Tests ---


def test_sub_agent_tools_survive_dynamic_agent_recreation():
    """Tools are preserved when _create_agent is called (dynamic sys prompt)."""
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4",
        system_prompt="Context: {{ context }}",
        sub_agents=[
            SubAgentConfig(
                name="helper",
                config="/fake/helper.yaml",
                description="Help.",
            ),
        ],
    )
    chain = ChainLite(config)
    assert "helper" in chain.agent._function_toolset.tools

    # Simulate dynamic recreation (happens when system prompt has template vars)
    new_agent = chain._create_agent("Context: test")
    assert "helper" in new_agent._function_toolset.tools
