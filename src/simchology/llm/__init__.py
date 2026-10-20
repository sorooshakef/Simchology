"""Shared machinery of the prompt-based methods: prompts, response parsing, caching,
network policy and the Ollama backend."""

from .cache import ResponseCache
from .netpolicy import enforce_remote_policy, is_local_url

__all__ = ["ResponseCache", "enforce_remote_policy", "is_local_url"]
