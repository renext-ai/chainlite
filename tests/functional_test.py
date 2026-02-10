import os
import pytest
from chainlite import ChainLite, ChainLiteConfig

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

test_config = ChainLiteConfig(
    llm_model_name="openai:gpt-3.5-turbo",
    prompt="Tell me a short, one-sentence joke about a computer.",
    model_settings={"temperature": 0.2},
    config_name="functional_test",
)


@pytest.fixture(autouse=True)
def check_env():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        pytest.skip("OPENAI_API_KEY is not configured; skipping integration tests.")


@pytest.fixture
def chain():
    return ChainLite(config=test_config)


def test_run(chain: ChainLite):
    response = chain.run({})
    assert response is not None
    assert len(str(response)) > 0


async def test_arun(chain: ChainLite):
    response = await chain.arun({})
    assert response is not None
    assert len(str(response)) > 0


def test_stream(chain: ChainLite):
    chunks = list(chain.stream({}))
    assert len(chunks) > 0
    assert len("".join(str(c) for c in chunks)) > 0


async def test_astream(chain: ChainLite):
    chunks = []
    async for chunk in chain.astream({}):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert len("".join(str(c) for c in chunks)) > 0


async def test_structured_output():
    structured_config = ChainLiteConfig(
        llm_model_name="openai:gpt-3.5-turbo",
        prompt="Tell me a joke about {{ topic }}",
        temperature=0.7,
        output_parser=[
            {"setup": "The setup of the joke"},
            {"punchline": "The punchline of the joke"},
        ],
        config_name="structured_test",
    )
    chain = ChainLite(config=structured_config)
    response = await chain.arun({"topic": "programming"})
    assert isinstance(response, dict)
    assert "setup" in response
    assert "punchline" in response
