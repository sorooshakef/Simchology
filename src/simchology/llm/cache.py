"""An append-only JSONL cache of model responses, so long runs can be resumed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._utils.hashing import stable_hash


class ResponseCache:
    """Remember responses by a stable key; optionally persist them to ``cache_dir``.

    With ``cache_dir=None`` the cache lives in memory only (it still avoids asking the
    same question twice within a session). With a directory, every response is appended
    to ``<cache_dir>/<name>.jsonl`` immediately, so an interrupted run over thousands of
    pairs continues where it stopped.
    """

    def __init__(self, cache_dir: str | Path | None = None, name: str = "responses") -> None:
        self._memory: dict[str, Any] = {}
        self.path: Path | None = None
        if cache_dir is not None:
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"{name}.jsonl"
            self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # a partially written last line
                    continue
                self._memory[record["key"]] = record["value"]

    @staticmethod
    def key(**parts: Any) -> str:
        return stable_hash(parts, length=32)

    def __contains__(self, key: str) -> bool:
        return key in self._memory

    def __len__(self) -> int:
        return len(self._memory)

    def get(self, key: str, default: Any = None) -> Any:
        return self._memory.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self._memory[key] = value
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
