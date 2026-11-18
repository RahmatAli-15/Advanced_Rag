"""Reranker interface. A cross-encoder reads query and passage together, so it judges
relevance far better than comparing two independently computed vectors, but it is too
slow to run over the whole corpus. Hence: retrieve broadly, then rerank a short list.
"""

import re
from typing import Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class OverlapReranker:
    """Word-overlap scorer for tests and offline development."""

    name = "test/overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z0-9]+", p.lower()))) / (len(q) or 1) for p in passages]
