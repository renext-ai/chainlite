from pathlib import Path


def test_pydantic_ai_private_api_usage_is_adapter_scoped():
    """Keep private pydantic-ai compatibility shims isolated in one adapter module."""
    root = Path(__file__).resolve().parents[1]
    adapter_path = root / "chainlite" / "adapters" / "pydantic_ai.py"

    patterns = [
        "_function_toolset",
        "_function_tools",
        "._instructions",
        "pydantic_ai.agent",
        "CallToolsNode",
    ]

    offenders = []
    for path in (root / "chainlite").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in text and path != adapter_path:
                offenders.append((str(path), pattern))

    assert not offenders, f"Private pydantic-ai usage found outside adapter: {offenders}"
