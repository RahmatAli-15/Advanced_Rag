"""Local embedding and reranking models via sentence-transformers (optional `ml` extra).

Models load lazily on first use, in fp16 on CUDA when available. Inference is
synchronous; async callers run it in a worker thread (asyncio.to_thread) so the event
loop keeps serving requests while the GPU works.
"""

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


def _device(preference: str) -> str:
    import torch

    if preference != "auto":
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        dim: int,
        device: str = "auto",
        batch_size: int = 16,
        query_prefix: str = "",
    ) -> None:
        self.name = model_name
        self.dim = dim
        self._device_pref = device
        self._batch_size = batch_size
        self._query_prefix = query_prefix  # bge-m3 needs none; bge-*-en-v1.5 does
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                device = _device(self._device_pref)
                model = SentenceTransformer(self.name, device=device)
                if device == "cuda":
                    model.half()  # halves VRAM; negligible quality loss for retrieval
                actual = model.get_embedding_dimension()
                if actual != self.dim:
                    raise ValueError(
                        f"{self.name} produces {actual}-d vectors, expected {self.dim}"
                    )
                log.info("loaded embedder %s on %s", self.name, device)
                self._model = model
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # unit vectors: cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.astype("float32").tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]


class CrossEncoderReranker:
    def __init__(
        self, model_name: str, device: str = "auto", batch_size: int = 16, max_length: int = 512
    ) -> None:
        self.name = model_name
        self._device_pref = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                device = _device(self._device_pref)
                model = CrossEncoder(self.name, device=device, max_length=self._max_length)
                if device == "cuda":
                    model.model.half()
                log.info("loaded reranker %s on %s", self.name, device)
                self._model = model
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._load().predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            activation_fn=None,
        )
        return [float(s) for s in scores]
