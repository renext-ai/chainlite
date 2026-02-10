# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- GitHub Actions CI workflow for unit tests across Python 3.10-3.12.
- GitHub Actions release workflow for building and publishing distributions.
- Release Please automation for SemVer versioning and changelog generation.
- Semantic PR title validation workflow.
- Contribution and security documentation.
- Pytest marker strategy for integration tests.

### Changed
- Package metadata improvements in `pyproject.toml`.
- README installation/testing instructions aligned with actual project setup.
- Integration tests now skip cleanly when credentials are missing.
