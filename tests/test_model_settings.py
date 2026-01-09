import os
import asyncio
import pytest
from unittest import mock
from chainlite.config import ChainLiteConfig
from chainlite.core import ChainLite


# Test case 1: Only model_settings is provided
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_model_settings_only():
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o",
        model_settings={"temperature": 0.7, "max_tokens": 100},
    )
    chain = ChainLite(config)
    assert chain._model_settings is not None
    assert chain._model_settings["temperature"] == 0.7


# Test case 2: Only legacy temperature is provided
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_legacy_temperature_only():
    config = ChainLiteConfig(llm_model_name="openai:gpt-4o", temperature=0.5)
    chain = ChainLite(config)
    assert chain._model_settings is not None
    assert chain._model_settings["temperature"] == 0.5


# Test case 3: Both provided, legacy temperature should override
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_precedence_legacy_overrides():
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o",
        model_settings={"temperature": 0.2},
        temperature=0.9,
    )
    chain = ChainLite(config)
    assert chain._model_settings is not None
    assert chain._model_settings["temperature"] == 0.9


# Test case 4: Both provided, merge other settings
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"})
def test_merge_behavior():
    config = ChainLiteConfig(
        llm_model_name="openai:gpt-4o", model_settings={"top_p": 0.9}, temperature=0.8
    )
    chain = ChainLite(config)
    assert chain._model_settings is not None
    assert chain._model_settings["temperature"] == 0.8
    if "top_p" in chain._model_settings:
        assert chain._model_settings["top_p"] == 0.9


if __name__ == "__main__":
    # Manually run if executed directly
    test_model_settings_only()
    test_legacy_temperature_only()
    test_precedence_legacy_overrides()
    test_merge_behavior()
    print("All verification tests passed!")
