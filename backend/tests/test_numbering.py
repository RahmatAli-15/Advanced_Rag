import pytest

from app.chunking.numbering import NumberingTracker, detect_marker, render_ref


@pytest.mark.parametrize(
    ("text", "style", "label"),
    [
        ("5. Definitions", "decimal", "5"),
        ("138A. Something", "decimal", "138A"),
        ("5.1 Dotted", "decimal", "5.1"),
        ("(2) item", "paren_decimal", "2"),
        ("(iv) item", "roman", "iv"),
        ("(a) item", "alpha", "a"),
        ("[ (2) amended text", "paren_decimal", "2"),
    ],
)
def test_detect_marker(text: str, style: str, label: str) -> None:
    m = detect_marker(text)
    assert m is not None and (m.style, m.label) == (style, label)


@pytest.mark.parametrize("text", ["Explanation: x", "Provided that", "5.5 per cent of", "Note:"])
def test_unnumbered_text_has_no_marker(text: str) -> None:
    # "5.5 per cent of ..." is a quantity, not clause 5.5
    assert detect_marker(text) is None


def _walk(lines: list[str]) -> list[str]:
    t = NumberingTracker()
    out = []
    for line in lines:
        m = t.advance(line)
        out.append(render_ref(t.path) if m else "-")
    return out


def test_hierarchy_is_learned_from_order() -> None:
    assert _walk(["5. x", "(1) a", "(i) b", "(ii) c", "(a) d", "(iii) e", "(2) f", "6. g"]) == [
        "5",
        "5(1)",
        "5(1)(i)",
        "5(1)(ii)",
        "5(1)(ii)(a)",
        "5(1)(iii)",
        "5(2)",
        "6",
    ]


def test_definitions_style_order_roman_then_alpha() -> None:
    assert _walk(["(iv) Beneficial Owner", "(a) company", "(b) firm", "(v) next"]) == [
        "(iv)",
        "(iv)(a)",
        "(iv)(b)",
        "(v)",
    ]


def test_i_after_h_is_a_letter_not_roman_one() -> None:
    assert _walk(["3. x", "(g) a", "(h) b", "(i) c", "(j) d"]) == [
        "3",
        "3(g)",
        "3(h)",
        "3(i)",
        "3(j)",
    ]


def test_unnumbered_lines_do_not_move_position() -> None:
    assert _walk(["7. x", "(1) a", "Explanation: b", "(2) c"]) == ["7", "7(1)", "-", "7(2)"]
