import pytest
import os
from chainlite import ChainLite, ChainLiteConfig
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def set_dummy_openai_key():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")


@pytest.mark.asyncio
async def test_jinja2_basic_interpolation():
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-3.5-turbo",
        prompt="Hello {{ name }}!",
    )
    chain = ChainLite(config)
    # White-box testing _build_prompt to avoid making API calls
    prompt = await chain._build_prompt({"name": "World"})
    assert "Hello World!" in str(prompt)


@pytest.mark.asyncio
async def test_jinja2_conditional():
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-3.5-turbo",
        prompt="{% if is_formal %}Good evening, {{ name }}.{% else %}Hi {{ name }}!{% endif %}",
    )
    chain = ChainLite(config)

    prompt_formal = await chain._build_prompt({"name": "Sir", "is_formal": True})
    assert "Good evening, Sir." in str(prompt_formal)

    prompt_informal = await chain._build_prompt({"name": "buddy", "is_formal": False})
    assert "Hi buddy!" in str(prompt_informal)


@pytest.mark.asyncio
async def test_jinja2_loop():
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-3.5-turbo",
        prompt="Items: {% for item in items %}- {{ item }}\n{% endfor %}",
    )
    chain = ChainLite(config)

    prompt = await chain._build_prompt({"items": ["Apple", "Banana"]})
    assert "- Apple" in str(prompt)
    assert "- Banana" in str(prompt)


@pytest.mark.asyncio
async def test_jinja2_missing_variable():
    # Jinja2 default behavior is to print empty string for undefined variables unless configured otherwise.
    # We want to ensure it doesn't crash, or if we want it to crash, we check for that.
    # Current implementation uses default environment, which renders undefined as empty.
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-3.5-turbo",
        prompt="Hello {{ name }}!",
    )
    chain = ChainLite(config)

    # Passing empty dict
    prompt = await chain._build_prompt({})
    assert "Hello !" in str(prompt)
    # Notes: The previous implementation raised KeyError.
    # If strictness is required, we might need to change implementation.
    # But for now, let's verify the current behavior.


def test_parse_input_variables():
    prompt_template = (
        "Hello {{ name }}, check this {% if show_secret %}{{ secret }}{% endif %}"
    )
    variables = ChainLite.parse_input_variables_from_prompt(prompt_template)
    assert "name" in variables
    assert "show_secret" in variables
    assert "secret" in variables
