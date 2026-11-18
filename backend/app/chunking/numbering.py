"""Clause numbering in regulatory text.

RBI directions nest numbered items, but the order of styles varies by document:
    5.  (1)  (i)  (a)          one document
    (iv) (a)                   another (definitions list)
The tracker learns the hierarchy from the order styles appear: a style already on
the stack is a sibling (pop back to it), a new style is a child (push).

"(i)", "(v)", "(x)" are ambiguous: roman numerals, or letters after (h), (u), (w).
They are read as letters only when they continue an alphabetic sequence.
"""

import re
from dataclasses import dataclass
from typing import Literal

Style = Literal["decimal", "paren_decimal", "roman", "alpha"]

_DECIMAL = re.compile(r"^(\d{1,3}[A-Z]?)\.(?!\d)\s*")  # "5." "138A."  (not "5.1")
# "5.1 Applicability", "5.1.2 (a)". Must be followed by a capital or "(" so that
# "5.5 per cent of ..." at the start of a line is not mistaken for clause 5.5.
_DOTTED = re.compile(r"^(\d{1,3}(?:\.\d{1,3})+)\.?\s+(?=[A-Z(\[])")
_PAREN = re.compile(r"^\(([0-9]{1,3}[A-Z]?|[a-z]{1,5}|[A-Z])\)\s*")
_ROMAN = re.compile(r"^(?=[ivxl])(x{0,3}|xl|l)?(ix|iv|v?i{0,3})$")
_LEADING_BRACKET = re.compile(r"^\[\s*")  # "[" marks text changed by an amendment

TOP_LEVEL: frozenset[Style] = frozenset({"decimal"})


@dataclass(frozen=True)
class Marker:
    style: Style
    label: str  # "5", "5.1", "2", "iv", "a"

    def render(self) -> str:
        return self.label if self.style == "decimal" else f"({self.label})"


def _is_roman(s: str) -> bool:
    return bool(_ROMAN.match(s))


def detect_marker(text: str, previous_alpha: str | None = None) -> Marker | None:
    """Find the leading clause marker of a paragraph, if it has one."""
    t = _LEADING_BRACKET.sub("", text)
    if m := _DOTTED.match(t):
        return Marker("decimal", m.group(1))
    if m := _DECIMAL.match(t):
        return Marker("decimal", m.group(1))
    if m := _PAREN.match(t):
        label = m.group(1)
        if label[0].isdigit():
            return Marker("paren_decimal", label)
        if label.isupper():
            return Marker("alpha", label.lower())
        continues_alpha = (
            previous_alpha is not None and len(label) == 1 and ord(label) == ord(previous_alpha) + 1
        )
        if _is_roman(label) and not continues_alpha:
            return Marker("roman", label)
        if len(label) == 1:
            return Marker("alpha", label)
    return None


class NumberingTracker:
    """Keeps the current position in the clause hierarchy, e.g. 5 > (2) > (iv)."""

    def __init__(self) -> None:
        self._stack: list[Marker] = []

    def reset(self) -> None:
        self._stack.clear()

    @property
    def path(self) -> tuple[Marker, ...]:
        return tuple(self._stack)

    def _last_alpha(self) -> str | None:
        return next((m.label for m in reversed(self._stack) if m.style == "alpha"), None)

    def advance(self, text: str) -> Marker | None:
        """Update the position for a paragraph. Returns its marker, or None if unnumbered."""
        marker = detect_marker(text, self._last_alpha())
        if marker is None:
            return None
        if marker.style in TOP_LEVEL:
            self._stack = [marker]
            return marker
        for i, existing in enumerate(self._stack):
            if existing.style == marker.style:
                self._stack = [*self._stack[:i], marker]  # sibling
                return marker
        self._stack.append(marker)  # child
        return marker


def render_ref(path: tuple[Marker, ...]) -> str:
    """(5, (2), (iv)) -> '5(2)(iv)'."""
    return "".join(m.render() for m in path)
