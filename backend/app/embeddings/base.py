"""Embedding model interface.

Retrieval code depends on this protocol, never on a specific library, so a hosted
embedding API or a different local model is a new adapter, not a rewrite. Tests use
HashEmbedder: deterministic and dependency-free.
"""

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector. Shares words => similar vectors.

    Only for tests and offline development; it has no notion of meaning.
    """

    def __init__(self, dim: int = 1024, name: str = "test/hash") -> None:
        self.dim = dim
        self.name = name

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
