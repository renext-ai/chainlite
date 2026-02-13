"""Prompt parsing/rendering helpers for ChainLite."""

from __future__ import annotations

from typing import List

import jinja2
from jinja2 import meta
from loguru import logger


def parse_input_variables_from_prompt(text: str) -> List[str]:
    """Parse jinja undeclared variables from a prompt template."""
    if not isinstance(text, str):
        return []

    env = jinja2.Environment()
    try:
        ast = env.parse(text)
        return list(meta.find_undeclared_variables(ast))
    except Exception as e:
        logger.error(f"Failed to parse input variables: {e}")
        return []
