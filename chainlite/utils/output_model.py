"""Output schema helper utilities for ChainLite."""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field, create_model


def merge_dictionaries(dict_list: List[Dict]) -> Dict:
    """Merge a list of dictionaries into a single dictionary."""
    merged_dict: Dict[str, Any] = {}
    for single_dict in dict_list:
        merged_dict.update(single_dict)
    return merged_dict


def get_type_from_string(type_str: str) -> Any:
    """Resolve a configured type string into a python type."""
    available_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "Dict[str, Any]": Dict[str, Any],
        "Dict[str, str]": Dict[str, str],
        "List[str]": List[str],
        "List[Dict[str, Any]]": List[Dict[str, Any]],
    }
    try:
        return available_types[type_str]
    except KeyError:
        logger.error(f"Unknown type: {type_str}")
        return str


def create_dynamic_pydantic_model(data: Dict[str, Any]) -> type[BaseModel]:
    """Create a dynamic pydantic model from dictionary metadata."""
    dynamic_model_fields = {}
    for key, value in data.items():
        if isinstance(value, str):
            dynamic_model_fields[key] = (str, Field(description=value))
            continue

        output_type = str
        output_description = None
        for v in value:
            if v.get("type"):
                output_type = get_type_from_string(v["type"])
            if v.get("description"):
                output_description = v["description"]

        dynamic_model_fields[key] = (
            output_type,
            Field(description=output_description) if output_description else ...,
        )

    return create_model("DynamicOutput", **dynamic_model_fields)
