import pytest

from app.retrieval.fusion import rrf_fuse
from app.retrieval.hybrid import keyword_tsquery


def test_item_in_both_lists_beats_item_in_one() -> None:
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]]))
    assert max(fused, key=lambda x: fused[x]) == "b"


def test_scores_follow_formula() -> None:
    fused = dict(rrf_fuse([["a"], ["a"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)


def test_order_is_by_score_and_handles_empty_lists() -> None:
    assert [x for x, _ in rrf_fuse([["a", "b"], []])] == ["a", "b"]
    assert rrf_fuse([[], []]) == []


def test_keyword_query_is_an_or_of_sanitised_words() -> None:
    assert keyword_tsquery("What's the V-CIP rule? NBFC & KYC!") == (
        "what | s | the | v | cip | rule | nbfc | kyc"
    )
    assert keyword_tsquery("?!") is None


def test_weighted_rrf_downweights_a_list() -> None:
    fused = rrf_fuse([["v1", "v2"], ["k1", "k2"]], weights=[1.0, 0.3])
    assert [x for x, _ in fused][:2] == ["v1", "v2"]
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[1.0, 2.0])
