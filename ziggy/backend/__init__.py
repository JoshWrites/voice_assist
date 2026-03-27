"""LLM backend abstraction.

Each backend (Ollama, Msty, future others) implements the same interface.
The factory auto-detects which is running and returns the appropriate client.
"""

import abc
from typing import Optional


class LLMBackend(abc.ABC):
    """Base class for LLM backends."""

    name: str = "base"
    url: str = ""

    @abc.abstractmethod
    def is_running(self) -> bool:
        """Check if this backend is reachable."""

    @abc.abstractmethod
    def start(self) -> bool:
        """Attempt to start this backend. Return True on success."""

    @abc.abstractmethod
    def get_default_model(self) -> Optional[str]:
        """Return the first available model name, or None."""

    @abc.abstractmethod
    def query(self, messages: list, model: str, temperature: float = 0.7,
              max_tokens: int = 500) -> Optional[str]:
        """Send a chat completion request. Return the response text or None."""


def detect_backend() -> Optional[LLMBackend]:
    """Auto-detect a running LLM backend. Returns the first one found."""
    from ziggy.backend.msty import MstyBackend
    from ziggy.backend.ollama import OllamaBackend

    for cls in [MstyBackend, OllamaBackend]:
        backend = cls()
        if backend.is_running():
            print(f"  Connected to {backend.name} backend")
            return backend
    return None
