from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, model_validator


CompactionMode = Literal["simple", "auto", "custom"]


class _BaseCompactionConfig(BaseModel):
    """Common compaction options for post-run and in-run flows."""

    model_config = ConfigDict(extra="forbid")

    mode: Optional[CompactionMode] = None
    truncation_threshold: int = 5000
    summarizor_config_path: Optional[str] = None
    summarizor_config_dict: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_custom_summarizor_source(self):
        if self.mode != "custom":
            return self

        if self.summarizor_config_path and self.summarizor_config_dict:
            raise ValueError(
                "Cannot specify both 'summarizor_config_path' and 'summarizor_config_dict'"
            )
        if not self.summarizor_config_path and not self.summarizor_config_dict:
            raise ValueError(
                "Must specify either 'summarizor_config_path' or 'summarizor_config_dict' for custom compaction"
            )
        return self


class PostRunCompactionConfig(_BaseCompactionConfig):
    """Post-run compaction configuration."""

    start_run: int = 1

    @model_validator(mode="after")
    def _normalize_start_run(self):
        self.start_run = max(1, int(self.start_run))
        return self


class InRunCompactionConfig(_BaseCompactionConfig):
    """In-run compaction configuration."""

    start_iter: int = 2
    start_run: int = 1
    max_concurrency: int = 4

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_keys(cls, data: Any):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "start_iter" not in normalized and "lazy_start_iter" in normalized:
            normalized["start_iter"] = normalized["lazy_start_iter"]
        if "start_run" not in normalized and "lazy_start_run" in normalized:
            normalized["start_run"] = normalized["lazy_start_run"]
        normalized.pop("lazy_start_iter", None)
        normalized.pop("lazy_start_run", None)
        return normalized

    @model_validator(mode="after")
    def _normalize_values(self):
        self.start_iter = max(1, int(self.start_iter))
        self.start_run = max(1, int(self.start_run))
        self.max_concurrency = max(1, int(self.max_concurrency))
        return self


class HistoryTruncatorConfig(BaseModel):
    """History truncation settings with backward compatibility normalization."""

    model_config = ConfigDict(extra="forbid")

    post_run_compaction: Optional[PostRunCompactionConfig] = None
    in_run_compaction: Optional[InRunCompactionConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_shape(cls, data: Any):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        root_post_keys = {
            "mode",
            "truncation_threshold",
            "start_run",
            "summarizor_config_path",
            "summarizor_config_dict",
        }
        has_flat_post = any(key in normalized for key in root_post_keys)
        if "post_run_compaction" not in normalized and has_flat_post:
            normalized["post_run_compaction"] = {
                key: normalized[key] for key in root_post_keys if key in normalized
            }

        if "in_run_compaction" not in normalized and isinstance(
            normalized.get("lazy_summarization"), dict
        ):
            normalized["in_run_compaction"] = normalized["lazy_summarization"]

        for key in root_post_keys:
            normalized.pop(key, None)
        normalized.pop("lazy_summarization", None)
        return normalized


class ChainLiteConfig(BaseModel):
    """
    Configuration model for ChainLite.
    """

    model_config = ConfigDict(extra="forbid")

    config_name: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    llm_model_name: str
    model_settings: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    output_parser: Optional[List[Dict[str, Any]]] = None
    use_history: Optional[bool] = False
    session_id: str = "unused"
    redis_url: Optional[str] = None
    max_retries: Optional[int] = 1
    history_truncator_config: Optional[HistoryTruncatorConfig] = None
