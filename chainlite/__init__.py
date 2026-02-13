try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError

        __version__ = version("chainlite")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"

"""
ChainLite - A lightweight wrapper for Pydantic AI.

This package provides a simple interface for building LLM-powered applications
with support for streaming, structured output, and conversation history.
"""

from .config import ChainLiteConfig
from .streaming import stream_sse

__all__ = ["ChainLite", "ChainLiteConfig", "stream_sse"]


def __getattr__(name: str):
    """Lazy-load heavy runtime dependencies for package-level imports."""
    if name == "ChainLite":
        from .core import ChainLite

        return ChainLite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
