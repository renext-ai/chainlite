# Contributing to ChainLite

## Development Setup

1. Create and activate a virtual environment.
2. Install package + dev dependencies:

```bash
pip install -e ".[dev]"
```

## Test Strategy

- Unit tests (default): do not require provider credentials.
- Integration tests: require a valid `OPENAI_API_KEY` and call real providers.

Run unit tests:

```bash
pytest
```

Run integration tests explicitly:

```bash
OPENAI_API_KEY=... pytest -m integration
```

## Pull Request Expectations

- Keep changes scoped and include tests for behavior changes.
- Update docs if public API or workflows change.
- Ensure `pytest` passes locally before opening a PR.
