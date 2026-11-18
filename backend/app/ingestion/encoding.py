"""Defensive decoding for government web pages.

Pages are decoded as UTF-8. Older Indian government sites sometimes mix in raw
Windows-1252 bytes (smart quotes, dashes) while declaring UTF-8; with
`errors="replace"` those silently become U+FFFD and corrupt titles. Instead, only
the invalid bytes are re-interpreted as Windows-1252, and valid UTF-8 is untouched.
"""

import codecs

_HANDLER = "utf8_cp1252_fallback"


def _cp1252_fallback(err: UnicodeError) -> tuple[str, int]:
    if not isinstance(err, UnicodeDecodeError):
        raise err
    bad = err.object[err.start : err.end]
    return bad.decode("cp1252", errors="replace"), err.end


codecs.register_error(_HANDLER, _cp1252_fallback)


def decode_html(raw: bytes) -> str:
    return raw.decode("utf-8", errors=_HANDLER)
