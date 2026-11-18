from app.chunking.chunker import ChunkConfig, chunk_blocks, resolve_amendments
from app.ingestion.document import Block, Footnote


def H(text: str) -> Block:
    return Block("heading", text)


def P(text: str, *footnotes: str) -> Block:
    return Block("paragraph", text, tuple(footnotes))


def words(n: int, word: str = "word") -> str:
    return " ".join([word] * n)


def test_clause_with_subitems_and_explanation_is_one_chunk() -> None:
    blocks = [
        H("Chapter II - Lending"),
        H("A. Fees"),
        P("5. A lender shall disclose all fees to the borrower before sanction."),
        P("(1) The disclosure shall be in writing."),
        P("Explanation: writing includes electronic form."),
        P("(2) Fees not disclosed shall not be charged."),
        P("6. Penal charges shall be reasonable and disclosed in the key facts statement."),
    ]
    chunks = chunk_blocks(blocks, ChunkConfig(min_tokens=1))

    assert [c.para_refs for c in chunks] == [("5", "5(1)", "5(2)"), ("6",)]
    first = chunks[0]
    assert first.heading_path == ("Chapter II - Lending", "A. Fees")
    assert "Explanation: writing includes electronic form." in first.text
    assert first.lead_in is None


def test_oversized_clause_splits_at_items_and_repeats_ancestor_chain() -> None:
    blocks = [
        P("5. In these Directions the following terms are defined:"),
        P("(1) Terms defined here:"),
        P("(xiv) 'Officially Valid Document' means any of the following:"),
        P("(a) " + words(120, "passport")),
        P("(b) " + words(120, "licence")),
        P("(c) " + words(120, "aadhaar")),
    ]
    chunks = chunk_blocks(blocks, ChunkConfig(max_tokens=200, min_tokens=1, lead_in_tokens=60))

    assert len(chunks) >= 2
    cont = chunks[1]
    assert cont.para_refs[0].startswith("5(1)(xiv)(")
    assert cont.lead_in is not None
    assert cont.lead_in.startswith("5. In these Directions")
    assert "Officially Valid Document" in cont.lead_in  # the parent that gives (b) meaning
    assert all(c.token_count <= 200 for c in chunks)


def test_numbered_heading_prefixes_subitem_refs() -> None:
    blocks = [H("63. CDD Procedure"), P("(9) For the purpose of KYC " + words(30))]
    (chunk,) = chunk_blocks(blocks, ChunkConfig(min_tokens=1))
    assert chunk.para_refs == ("63(9)",)


def test_tiny_clauses_in_a_section_are_merged() -> None:
    blocks = [H("A. Short title"), P("1. These are the Directions."), P("2. In force now.")]
    (chunk,) = chunk_blocks(blocks, ChunkConfig(min_tokens=60))
    assert chunk.para_refs == ("1", "2")


def test_merging_never_crosses_headings() -> None:
    blocks = [H("A. One"), P("1. Short."), H("B. Two"), P("2. Short.")]
    chunks = chunk_blocks(blocks, ChunkConfig(min_tokens=60))
    assert [c.heading_path for c in chunks] == [("A. One",), ("B. Two",)]


def test_huge_sentence_without_breaks_is_split_by_words() -> None:
    blocks = [P("7. " + ", ".join(["the Some Act 1934"] * 400))]
    chunks = chunk_blocks(blocks, ChunkConfig(max_tokens=100, min_tokens=1))
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_large_table_is_split_by_rows_with_header_repeated() -> None:
    rows = "\n".join(f"Item {i} | {words(20)}" for i in range(30))
    blocks = [Block("table", "Item | Description\n" + rows)]
    chunks = chunk_blocks(blocks, ChunkConfig(max_tokens=150, min_tokens=1))
    assert len(chunks) > 1
    assert all(c.kind == "table" and c.text.startswith("Item | Description") for c in chunks)


def test_no_text_is_lost() -> None:
    blocks = [
        H("Chapter I"),
        P("1. " + words(50, "alpha")),
        P("(1) " + words(300, "beta")),
        P("(2) " + words(10, "gamma")),
        Block("table", "A | B\n" + "\n".join(f"{i} | {words(30, 'delta')}" for i in range(20))),
        P("2. " + words(5, "epsilon")),
    ]
    chunks = chunk_blocks(blocks, ChunkConfig(max_tokens=120, min_tokens=10))
    out = " ".join(c.text for c in chunks).split()
    for w in ("alpha", "beta", "gamma", "delta", "epsilon"):
        src = sum(b.text.split().count(w) for b in blocks)
        assert out.count(w) >= src, w  # >= because sentence overlap may repeat text


def test_amendments_are_resolved_for_citation() -> None:
    from datetime import date

    blocks = [P("5. Text " + words(20), "F1")]
    (chunk,) = chunk_blocks(blocks, ChunkConfig(min_tokens=1))
    footnotes = {
        "F1": Footnote("F1", "Substituted ...", "substituted", date(2026, 7, 1), "X", "https://u")
    }
    assert resolve_amendments(chunk, footnotes) == [
        {
            "id": "F1",
            "action": "substituted",
            "effective_from": "2026-07-01",
            "amending_title": "X",
            "amending_url": "https://u",
            "text": "Substituted ...",
        }
    ]


def test_context_header() -> None:
    blocks = [H("Chapter I"), P("3. Applicability " + words(30))]
    (chunk,) = chunk_blocks(blocks, ChunkConfig(min_tokens=1))
    assert chunk.context_header("KYC Directions") == "KYC Directions > Chapter I > Para 3"
