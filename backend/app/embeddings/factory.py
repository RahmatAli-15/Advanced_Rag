"""Build models from settings. The only place that knows which implementation is used."""

from app.core.config import Settings
from app.db.models import EMBEDDING_DIM
from app.embeddings.base import Embedder, HashEmbedder
from app.retrieval.rerank import OverlapReranker, Reranker


def build_embedder(settings: Settings) -> Embedder:
    if settings.models_backend == "fake":
        return HashEmbedder(dim=EMBEDDING_DIM)
    from app.embeddings.local import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        settings.embedding_model, dim=EMBEDDING_DIM, device=settings.ml_device
    )


def build_reranker(settings: Settings) -> Reranker:
    if settings.models_backend == "fake":
        return OverlapReranker()
    from app.embeddings.local import CrossEncoderReranker

    return CrossEncoderReranker(settings.reranker_model, device=settings.ml_device)
