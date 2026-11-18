"""Reciprocal Rank Fusion (Cormack et al., 2009).

Vector and keyword search return scores on unrelated scales (cosine similarity vs
ts_rank), so the scores cannot be added. RRF uses only ranks:

    score(d) = sum over lists of 1 / (k + rank_in_list(d))

k = 60 is the value from the paper; it damps the advantage of rank 1 over rank 2 so
one list cannot dominate.

Weighted RRF multiplies each list's contribution by a weight, for when one retriever
is measurably less reliable than the other (see eval/reports).
"""

from collections.abc import Hashable, Sequence

RRF_K = 60


def rrf_fuse[T: Hashable](
    ranked_lists: Sequence[Sequence[T]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[T, float]]:
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    scores: dict[T, float] = {}
    for i, ranking in enumerate(ranked_lists):
        w = 1.0 if weights is None else weights[i]
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
