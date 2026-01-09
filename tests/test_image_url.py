import os
import tempfile
from unittest import mock

# import pytest removed
from chainlite import ChainLite, ChainLiteConfig
from pydantic_ai import ImageUrl, BinaryContent


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_build_prompt_text_only():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Hello {{input}}"
    )
    chain = ChainLite(config)
    input_data = {"input": "World"}

    prompt = chain._build_prompt(input_data)
    assert prompt == "Hello World"
    assert isinstance(prompt, str)


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_build_prompt_with_image_url():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Describe {{input}}"
    )
    chain = ChainLite(config)
    input_data = {"input": "this image", "image_url": "https://example.com/test.png"}

    prompt = chain._build_prompt(input_data)
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0] == "Describe this image"
    assert isinstance(prompt[1], ImageUrl)
    assert prompt[1].url == "https://example.com/test.png"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_build_prompt_with_empty_image_url():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Hello {{input}}"
    )
    chain = ChainLite(config)
    input_data = {"input": "World", "image_url": None}

    prompt = chain._build_prompt(input_data)
    assert prompt == "Hello World"
    assert isinstance(prompt, str)


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_build_prompt_with_local_file():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Analyze {{input}}"
    )
    chain = ChainLite(config)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"fake image content")
        tmp_path = tmp.name

    try:
        input_data = {"input": "file", "image_url": tmp_path}

        prompt = chain._build_prompt(input_data)
        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0] == "Analyze file"
        # Since it's local file, it should be BinaryContent
        assert isinstance(prompt[1], BinaryContent)
        assert prompt[1].data == b"fake image content"
        # .txt usually mimics text/plain, but let's see what mimetypes guesses or if it falls back.
        # Strict checking mime might be flaky across OS, but let's check it's not None.
        assert prompt[1].media_type is not None

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    try:
        test_build_prompt_text_only()
        test_build_prompt_with_image_url()
        test_build_prompt_with_empty_image_url()
        test_build_prompt_with_local_file()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
