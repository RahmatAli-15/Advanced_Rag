from app.eval.dataset import GoldItem
from app.eval.generate import copies_clause
from app.eval.metrics import RetrievedRef, first_doc_rank, first_match_rank, score

GOLD = GoldItem(
    id="g1",
    question="q",
    language="en",
    doc_source_key="kyc",
    doc_title="KYC",
    para_refs=["42(1)", "42(2)"],
    text_snippet="(1) The NBFC shall adopt a risk-based approach",
    origin="test",
)


def test_match_requires_same_document_and_ref_or_snippet() -> None:
    hits = [
        RetrievedRef("other", ["42(1)"], "x"),  # same ref, wrong document
        RetrievedRef("kyc", ["7"], "unrelated"),  # right document, wrong clause
        RetrievedRef("kyc", ["42(2)", "42(3)"], "..."),  # shares a ref
    ]
    assert first_match_rank(hits, GOLD) == 3
    assert first_doc_rank(hits, GOLD) == 2


def test_snippet_match_survives_rechunking() -> None:
    hits = [
        RetrievedRef(
            "kyc", ["41(9)", "42"], "... (1) The NBFC shall adopt a risk-based approach for"
        )
    ]
    assert first_match_rank(hits, GOLD) == 1


def test_score_rates_and_mrr() -> None:
    s = score([1, 3, None, 12], [1, 1, 2, None], [10, 20, 30, 40], ["a", "b", "c", "d"])
    assert (s.hit1, s.hit5, s.hit10) == (0.25, 0.5, 0.5)
    assert s.mrr10 == round((1 + 1 / 3) / 4, 3)
    assert s.doc5 == 0.75
    assert s.misses == ["c", "d"]


def test_copy_check() -> None:
    clause = "The NBFC shall intimate its customers in advance to update their KYC records."
    assert copies_clause("Must the NBFC intimate its customers in advance before KYC?", clause)
    assert not copies_clause("Do lenders need to warn borrowers before KYC refresh is due?", clause)
