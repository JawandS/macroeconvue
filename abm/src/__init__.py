"""MacroEconVue ABM top-level package."""

from importlib import import_module
from typing import Any

# Lazily expose key subpackages to provide clean imports when running main directly.

def __getattr__(name: str) -> Any:
    if name in {"config", "agents", "services"}:
        return import_module(f"src.{name}")
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = ["config", "agents", "services"]
