# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0](https://github.com/renext-ai/chainlite/compare/chainlite-v0.1.0...chainlite-v0.2.0) (2026-02-10)


### Features

* Add `StreamProcessor` to enhance streaming of structured model outputs, such as intermediate thoughts, and introduce a synchronous chat test. ([c36fb20](https://github.com/renext-ai/chainlite/commit/c36fb2020088ac096a6723e63361320846920674))
* Add conditional streaming logic to use `stream_text` as a fallback when `stream_responses` is unavailable. ([667aad0](https://github.com/renext-ai/chainlite/commit/667aad058411f4d30aed43fc707c74a5cfcd77cb))
* add jinja2 support ([cc8a46c](https://github.com/renext-ai/chainlite/commit/cc8a46c6772f7ef13baa79584b25c20b1a71a492))
* Add support for streaming structured results from processor outputs. ([6de467d](https://github.com/renext-ai/chainlite/commit/6de467d474b1fa7556154e5b5c7f8ff9ada089a0))
* Implement dynamic system prompts using Jinja2 templating based on input context. ([70088e1](https://github.com/renext-ai/chainlite/commit/70088e159f8f9a18737ce6cabbb6491b4f0f79ce))
* inherit existing agent's tools when creating a new agent instance. ([6ed0676](https://github.com/renext-ai/chainlite/commit/6ed0676ea349f16b11b076768c4ce9a20cf134dd))
* Set up pytest for integration tests, add GitHub Actions CI/CD workflows, and introduce project documentation. ([0f81c4c](https://github.com/renext-ai/chainlite/commit/0f81c4c01671f8b39a37b5a03bf63160821bfbab))
* Smart History Truncation ([b232bde](https://github.com/renext-ai/chainlite/commit/b232bde05e375f1c9246cd25c971f9d6809da7e4))
* support pydantic-ai deps ([5b8af1a](https://github.com/renext-ai/chainlite/commit/5b8af1ace420e96070e54e688d81b7f7adbe382e))


### Bug Fixes

* update pyproject dependencies ([0da72a9](https://github.com/renext-ai/chainlite/commit/0da72a953b2605d3b8960d518146dabd222b21c7))
* update the provider resolver ([0ec628a](https://github.com/renext-ai/chainlite/commit/0ec628ac906469fabfe675bbc684af8cbf814851))


### Documentation

* add an example of chatting with tools and dependencies ([b9c1c81](https://github.com/renext-ai/chainlite/commit/b9c1c81b185a9b7231390b641e3b19422aafb874))
* add structured output description ([bdeb4ea](https://github.com/renext-ai/chainlite/commit/bdeb4ea530dc1f01f5d4a3fa0c7f5ee70710d4d6))
* add tools docs ([77137d1](https://github.com/renext-ai/chainlite/commit/77137d13479afb21726248103deebb4ae6d0e337))
* fix incorrect sample code ([791b17d](https://github.com/renext-ai/chainlite/commit/791b17da57b4bb316836b1b57d146533500f91a0))
* update docs ([1149bc6](https://github.com/renext-ai/chainlite/commit/1149bc660ee4cbf0627aaa15d71c74aa3d349f30))
* update docs ([ce4657f](https://github.com/renext-ai/chainlite/commit/ce4657f13cbd9fcfa55c211589ddce7807a269d2))

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
