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
- Use a semantic PR title: `feat: ...`, `fix: ...`, `docs: ...`, etc.

## Versioning and Releases

This project uses SemVer and release automation.

- `fix` -> PATCH release
- `feat` -> MINOR release
- Any breaking change (`!` in title, e.g. `feat!: ...`) -> MAJOR release

Release flow:

1. Merge PRs to `main`.
2. Release Please opens/updates a release PR with version and changelog updates.
3. Merge the release PR to create a GitHub Release and tag.
4. The publish workflow builds and uploads package artifacts to PyPI (only when repository variable `PYPI_PUBLISH_ENABLED=true`).
