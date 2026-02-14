# Changelog

All notable changes to this project will be documented in this file.

## [0.5.5](https://github.com/renext-ai/chainlite/compare/chainlite-v0.5.4...chainlite-v0.5.5) (2026-02-14)


### Bug Fixes

* Scope in-run compaction to only process messages from the current run. ([34047cd](https://github.com/renext-ai/chainlite/commit/34047cd7fc2d97c7b637613423eba1dbaef2d330))

## [0.5.4](https://github.com/renext-ai/chainlite/compare/chainlite-v0.5.3...chainlite-v0.5.4) (2026-02-14)


### Features

* Add runtime guards for sync API calls and refine in-run compaction configuration with type safety and validation. ([4ff95af](https://github.com/renext-ai/chainlite/commit/4ff95afd4c1655a4c678c5b50fcb6105e390d742))


### Code Refactoring

* Extract `_ensure_no_running_loop` into a new `async_utils` module and rename it to `ensure_no_running_loop`. ([a217003](https://github.com/renext-ai/chainlite/commit/a21700382baa74a5ec59dd061151f0e260bdb571))

## [0.5.3](https://github.com/renext-ai/chainlite/compare/chainlite-v0.5.2...chainlite-v0.5.3) (2026-02-14)


### Features

* Preserve original messages in raw history while applying compaction to context history. ([8a12b0d](https://github.com/renext-ai/chainlite/commit/8a12b0dee9c631d9a290efdae05402669b094071))


### Code Refactoring

* Extract YAML configuration loading to a dedicated module and refine compaction manager access. ([44f9b80](https://github.com/renext-ai/chainlite/commit/44f9b804301ae6ed53d59182868ae9a62561b056))

## [0.5.2](https://github.com/renext-ai/chainlite/compare/chainlite-v0.5.1...chainlite-v0.5.2) (2026-02-13)


### Features

* **core:** add real-time in-run compaction for stream/astream ([b95e61b](https://github.com/renext-ai/chainlite/commit/b95e61b0a53d351a965c2f1480ad688cc0e2b685))


### Miscellaneous Chores

* trigger release 0.5.2 ([89e10be](https://github.com/renext-ai/chainlite/commit/89e10be0335160214bfa9e59aa556e96d115dce9))

## [0.5.1](https://github.com/renext-ai/chainlite/compare/chainlite-v0.5.0...chainlite-v0.5.1) (2026-02-13)


### Bug Fixes

* **core:** ensure stream/astream increment run_count and pass context for truncation ([452aec3](https://github.com/renext-ai/chainlite/commit/452aec3c43756c78ff91d6ca957b4b1b3875c673))
* **core:** ensure stream/astream increment run_count and pass context… ([70ee73d](https://github.com/renext-ai/chainlite/commit/70ee73d9a2b06efb88169e3ff5d6a8c0102df2a3))

## [0.5.0](https://github.com/renext-ai/chainlite/compare/chainlite-v0.4.1...chainlite-v0.5.0) (2026-02-13)


### Features

* add lazy trucators ([d8fd832](https://github.com/renext-ai/chainlite/commit/d8fd8323dd530940d0fcad65e5ce99a9b25c6466))
* add lazy trucators and pydantic-ai adapter ([ba99987](https://github.com/renext-ai/chainlite/commit/ba9998735c1a67bb55a6f99574ab29fc2d9c2e82))
* Implement token usage tracking and conduct experiments on history truncation strategies. ([c58f4bf](https://github.com/renext-ai/chainlite/commit/c58f4bf0a1d62c525223fc7fd36b07b90d9c8ce0))
* Introduce `get_agent_tool_schemas` for normalized tool schema extraction, enforce pydantic-ai adapter boundaries, and pin pydantic-ai dependencies to `1.49.0`. ([59674c4](https://github.com/renext-ai/chainlite/commit/59674c4c0101f791162af420141b773a6649a96f))
* Refactor token usage ablation study to track summarizer stats, rename compaction strategies, and introduce new utility modules. ([5bbee27](https://github.com/renext-ai/chainlite/commit/5bbee277781aef6a853884980fdac9f72114ca8d))


### Bug Fixes

* Corrected "summarizor" spelling to "summarizer" across classes, configurations, and documentation, adding backward compatibility for legacy keys. ([129edad](https://github.com/renext-ai/chainlite/commit/129edad6759176c5b1abb091472ea1c0d6228cbb))

## [0.4.1](https://github.com/renext-ai/chainlite/compare/chainlite-v0.4.0...chainlite-v0.4.1) (2026-02-12)


### Continuous Integration

* fix setuptools_scm version resolution in GitHub Actions ([9fc4f1a](https://github.com/renext-ai/chainlite/commit/9fc4f1a7ed4a694bde037b8329fc7bb7c55955d9))

## [0.4.0](https://github.com/renext-ai/chainlite/compare/chainlite-v0.3.0...chainlite-v0.4.0) (2026-02-12)


### Features

* enable dynamic versioning via setuptools_scm ([f444da3](https://github.com/renext-ai/chainlite/commit/f444da3c7de82aa0ef55393fc8ed0dc741646ba0))
* enhance history management with dynamic system prompt storage and improved markdown/JSON exports ([919868d](https://github.com/renext-ai/chainlite/commit/919868dde0ea531ce6497a48d08330db636c7e1d))

## [0.3.0](https://github.com/renext-ai/chainlite/compare/chainlite-v0.2.0...chainlite-v0.3.0) (2026-02-12)


### Features

* Enforce strict Pydantic model validation for `ChainLiteConfig` and update various dependencies. ([7d3a87d](https://github.com/renext-ai/chainlite/commit/7d3a87d482209821de9b2d0e0fe0ea03c2805fa9))
* feature/history truncation summarization ([7afb74e](https://github.com/renext-ai/chainlite/commit/7afb74e879b31e6acf1dba1b403f99d7268c4e92))
* History truncation, summarization, export, and config refactor ([0c4c345](https://github.com/renext-ai/chainlite/commit/0c4c345a6e8afc651e0604e30e8d164b50e53ee0))
* Implement pluggable history truncation with dual context/raw message storage and async message addition. ([73a3f12](https://github.com/renext-ai/chainlite/commit/73a3f1213251cd8f444a9af0dfb7dfc62ca4d9ff))


### Bug Fixes

* Resolve unit test failures for history export and truncation ([7bbf76c](https://github.com/renext-ai/chainlite/commit/7bbf76c0b740c56f01e0e2c7fbac5ed6a5510015))
* Use pathlib for history export and skip auto-summary test on dummy key ([3028bdd](https://github.com/renext-ai/chainlite/commit/3028bdd0e54f90e14cc3fb66080c238451c7d627))


### Documentation

* Delete debug sync stream script and add documentation for history management. ([a01c152](https://github.com/renext-ai/chainlite/commit/a01c15291a6dd8a96c812baecd40b9dc15ef8b22))

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
