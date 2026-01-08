import os
import tempfile
import base64
from unittest import mock
from chainlite import ChainLite, ChainLiteConfig
from pydantic_ai import ImageUrl, BinaryContent


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_multiple_images_urls():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Describe these images {input}"
    )
    chain = ChainLite(config)
    input_data = {
        "input": "",
        "images": ["https://example.com/1.png", "https://example.com/2.png"],
    }

    prompt = chain._build_prompt(input_data)
    assert isinstance(prompt, list)
    assert len(prompt) == 3
    assert prompt[0] == "Describe these images "
    assert isinstance(prompt[1], ImageUrl)
    assert prompt[1].url == "https://example.com/1.png"
    assert isinstance(prompt[2], ImageUrl)
    assert prompt[2].url == "https://example.com/2.png"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_base64_image():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Describe {input}"
    )
    chain = ChainLite(config)
    # Tiny base64 gif
    b64_str = (
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )

    input_data = {"input": "this base64 image", "images": [b64_str]}

    prompt = chain._build_prompt(input_data)
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].media_type == "image/gif"
    assert prompt[1].data == base64.b64decode(
        "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_local_and_remote_mix():
    config = ChainLiteConfig(
        llm_model_name="openai:test-model", prompt="Compare {input}"
    )
    chain = ChainLite(config)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"fake png")
        tmp_path = tmp.name

    try:
        input_data = {
            "input": "inputs",
            "images": [tmp_path, "https://example.com/remote.jpg"],
        }

        prompt = chain._build_prompt(input_data)
        assert len(prompt) == 3
        # First item is local file -> BinaryContent
        assert isinstance(prompt[1], BinaryContent)
        # Second item is remote url -> ImageUrl
        assert isinstance(prompt[2], ImageUrl)
        assert prompt[2].url == "https://example.com/remote.jpg"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_legacy_image_url():
    config = ChainLiteConfig(llm_model_name="openai:test-model")
    chain = ChainLite(config)

    input_data = {"input": "legacy", "image_url": "https://example.com/legacy.png"}

    prompt = chain._build_prompt(input_data)
    assert len(prompt) == 2
    assert isinstance(prompt[1], ImageUrl)
    assert prompt[1].url == "https://example.com/legacy.png"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_legacy_and_new_mix():
    # If both are present, they should be concatenated
    config = ChainLiteConfig(llm_model_name="openai:test-model")
    chain = ChainLite(config)

    input_data = {
        "input": "mix",
        "image_url": "https://example.com/legacy.png",
        "images": ["https://example.com/new.png"],
    }

    prompt = chain._build_prompt(input_data)
    # prompt string + new image + legacy image
    assert len(prompt) == 3
    # Order depends on implementation: images list first, then legacy
    assert prompt[1].url == "https://example.com/new.png"
    assert prompt[2].url == "https://example.com/legacy.png"


if __name__ == "__main__":
    try:
        test_multiple_images_urls()
        test_base64_image()
        test_local_and_remote_mix()
        test_legacy_image_url()
        test_legacy_and_new_mix()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
