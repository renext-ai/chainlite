"""
ChainLite - A lightweight wrapper for Pydantic AI.

This package provides a simple interface for building LLM-powered applications
with support for streaming, structured output, and conversation history.
"""

from .core import ChainLite
from .config import ChainLiteConfig
from .streaming import stream_sse

__all__ = ["ChainLite", "ChainLiteConfig", "stream_sse"]
