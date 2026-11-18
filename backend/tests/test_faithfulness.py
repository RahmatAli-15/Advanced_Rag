"""Answer evaluation: claim splitting, deterministic checks, planting, judging, scoring."""

import json
import random
import uuid
from datetime import date

from app.answer.service import Citation
from app.eval.dataset import GoldItem
from app.eval.faithfulness import (
    Claim,
    ItemResult,
    devanagari_share,
    judge_claims,
    language_ok,
    missing_numbers,
    numbers_in,
    plant_wrong_number,
    sample,
    split_claims,
    summarise,
)
from app.llm.gateway import LLMGateway, ModelRoute
from app.llm.types import LLMRequest, LLMResponse, Usage


def _src(cid: str, text: str, refs: list[str] | None = None) -> Citation:
    return Citation(
        id=cid,
        chunk_id=uuid.uuid4(),
        document_title="KYC Directions",
        para_refs=refs or ["42(1)"],
        source_url="https://e",
        version_valid_from=date(2025, 12, 29),
        amendments=[],
        quote=text[:400],
        text=text,
    )


KYC = _src(
    "S1",
    "The NBFC shall carry out periodic updation at least once in every two years for "
    "high-risk customers and once in every 10 years for low-risk customers.",
)
GOLD = _src("S2", "LTV shall not exceed 85% for loans up to ₹2.5 lakh.", ["43"])

# --- claims ---------------------------------------------------------------------------


def test_split_claims_keeps_citations_with_their_sentence() -> None:
    claims = split_claims(
        "High-risk customers: every **2 years** [S1]. Low-risk: every 10 years [S1][S2].\n"
        "The NBFC must notify customers first."
    )
    assert [(c.text, c.cited) for c in claims] == [
        ("High-risk customers: every 2 years .", ["S1"]),
        ("Low-risk: every 10 years .", ["S1", "S2"]),
        ("The NBFC must notify customers first.", []),
    ]


def test_split_claims_skips_headings_titles_and_fragments() -> None:
    # Taken from the first evaluation run, where these were judged as "unsupported".
    answer = (
        "**Answer (Hindi)**\n"
        "## Summary\n"
        "Summary\n"
        "Form Segment\n"
        "| Form | Segment |\n|---|---|\n| Form 1 | Consumer [S1] |\n"
        "**Deadline:** within 2 weeks [S2]."
    )
    texts = [c.text for c in split_claims(answer)]
    assert texts == ["Form 1 Consumer", "Deadline: within 2 weeks ."]


def test_split_claims_moves_leading_citations_back_and_handles_markdown() -> None:
    answer = (
        "Key limits:\n"
        "- LTV up to 85%. [S2] Higher loans get less.\n"
        "| Loan size | LTV |\n|---|---|\n| up to ₹2.5 lakh | 85% [S2] |\n"
        "हाई-रिस्क ग्राहकों के लिए हर दो साल। [S1]"
    )
    claims = split_claims(answer)
    texts = [(c.text, c.cited) for c in claims]
    assert ("LTV up to 85%.", ["S2"]) in texts  # citation moved back to its sentence
    assert ("Higher loans get less.", []) in texts
    assert ("up to ₹2.5 lakh 85%", ["S2"]) in texts  # table row is one claim
    assert ("हाई-रिस्क ग्राहकों के लिए हर दो साल।", ["S1"]) in texts
    assert all(not c.text.endswith(":") for c in claims)  # lead-ins are not claims
    assert [c.id for c in claims] == [f"C{i}" for i in range(1, len(claims) + 1)]


# --- deterministic checks -------------------------------------------------------------


def test_numbers_normalise_digits_words_and_commas() -> None:
    assert numbers_in("₹2,50,000 within two years, १० days, 85.0%") == {"250000", "2", "10", "85"}


def test_missing_numbers_flags_a_wrong_limit_but_accepts_words_and_refs() -> None:
    assert missing_numbers("High-risk: every 2 years (para 42(1))", [KYC]) == []
    assert missing_numbers("High-risk: every 3 years", [KYC]) == ["3"]
    assert missing_numbers("LTV up to 90% for ₹2.5 lakh", [GOLD]) == ["90"]


def test_language_match_by_script() -> None:
    assert devanagari_share("हर दो साल KYC") > 0.5
    assert language_ok("hi", "हर दो साल में KYC अपडेट करना होगा")
    assert not language_ok("hinglish", "हर दो साल में अपडेट करना होगा")  # observed failure
    assert language_ok("hinglish", "Har do saal mein KYC update karna hoga")
    assert language_ok("en", "Every two years.")


def test_sample_is_stratified_and_reproducible() -> None:
    items = [
        GoldItem(
            id=f"{i}-{lang}",
            question="q",
            language=lang,  # type: ignore[arg-type]
            doc_source_key="k",
            doc_title="t",
            para_refs=["1"],
            text_snippet="x",
            origin="test",
        )
        for i in range(10)
        for lang in ("en", "hinglish", "hi")
    ]
    a, b = sample(items, 6, seed=1), sample(items, 6, seed=1)
    assert [i.id for i in a] == [i.id for i in b]
    assert sorted(i.language for i in a) == ["en", "en", "hi", "hi", "hinglish", "hinglish"]


# --- planted claim --------------------------------------------------------------------


def test_planted_claim_changes_a_number_to_one_no_source_contains() -> None:
    claims = [Claim("C1", "LTV up to 85% for small loans", ["S2"])]
    planted = plant_wrong_number(claims, [GOLD], random.Random(0))
    assert planted is not None and planted.planted
    assert planted.text == "LTV up to 95% for small loans"
    assert planted.cited == ["S2"]


def test_no_plant_without_a_grounded_number() -> None:
    claims = [Claim("C1", "Customers must be told first", ["S1"])]
    assert plant_wrong_number(claims, [KYC], random.Random(0)) is None


# --- judge ----------------------------------------------------------------------------


class JudgeLLM:
    name = "fake"

    def __init__(self, reply: dict[str, object]) -> None:
        self.reply = reply
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest, model: str) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(json.dumps(self.reply), model, "fake", Usage(1, 1))


async def test_judge_is_blind_to_citations_and_fills_verdicts() -> None:
    llm = JudgeLLM(
        {
            "claims": [
                {"id": "C1", "verdict": "supported", "supported_by": ["S1"], "reason": "ok"},
                {"id": "C2", "verdict": "contradicted", "supported_by": [], "reason": "10 not 3"},
            ]
        }
    )
    gateway = LLMGateway([ModelRoute(llm, "judge", 6000)], max_retries=0)
    claims = [
        Claim("C1", "High-risk: every two years", ["S1"]),
        Claim("C2", "Low-risk: every 3 years", ["S1"]),
        Claim("C3", "Something else", []),
    ]
    await judge_claims(gateway, "KYC?", claims, [KYC])

    assert [c.verdict for c in claims] == ["supported", "contradicted", "missing"]
    user = llm.requests[0].messages[1].content
    assert "C2: Low-risk: every 3 years" in user
    assert "[S1]" not in user.split("Claims:")[1]  # the judge never sees the citations


# --- scoring --------------------------------------------------------------------------


def test_summary_separates_faithfulness_citation_accuracy_and_judge_sensitivity() -> None:
    ok = ItemResult("a", "answerable", "en", "q", "answered", gold_in_sources=True)
    ok.language_ok = True
    ok.claims = [
        Claim("C1", "every 2 years", ["S1"], verdict="supported", supported_by=["S1"]),
        Claim("C2", "every 10 years", ["S2"], verdict="supported", supported_by=["S1"]),
    ]
    ok.planted = Claim("C3", "every 12 years", ["S1"], planted=True, verdict="contradicted")
    bad = ItemResult("b", "answerable", "hinglish", "q", "answered", gold_in_sources=False)
    bad.language_ok = False
    bad.claims = [Claim("C1", "LTV 90%", ["S2"], verdict="contradicted", numbers_missing=["90"])]
    bad.planted = Claim("C2", "LTV 95%", ["S2"], planted=True, verdict="supported")
    abstained = ItemResult("c", "answerable", "hi", "q", "not_found", gold_in_sources=False)
    trap = ItemResult("u1", "unanswerable", "en", "SEBI?", "answered")
    declined = ItemResult("u2", "unanswerable", "en", "France?", "not_found")

    s = summarise([ok, bad, abstained, trap, declined])
    claims = s["claims"]
    assert isinstance(claims, dict)
    assert claims["supported"] == 0.667
    assert claims["citation_accurate"] == 0.333  # C2 cites S2, but S1 supports it
    assert claims["numbers_grounded"] == 0.667
    assert s["answers_fully_supported"] == 0.5
    assert s["language_ok"] == 0.5
    assert s["judge"] == {"planted": 2, "caught": 0.5, "errors": 0}
    assert s["unanswerable"] == {"n": 2, "abstained": 0.5, "answered_anyway": ["u1"]}
    assert s["answerable"] == {
        "n": 3,
        "answered": 0.667,
        "abstained": 0.333,
        "gold_in_sources": 0.333,
    }
