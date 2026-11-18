import hashlib
from pathlib import Path

from app.ingestion.raw_store import RawStore


def test_put_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    first = store.put(b"<html>v1</html>", ".html")
    again = store.put(b"<html>v1</html>", ".html")

    assert first.created and not again.created
    assert first.sha256 == hashlib.sha256(b"<html>v1</html>").hexdigest()
    assert first.relative_path == again.relative_path
    assert first.relative_path.startswith(first.sha256[:2] + "/")
    assert store.get(first.relative_path) == b"<html>v1</html>"


def test_different_content_gets_different_paths(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    assert store.put(b"a", ".html").relative_path != store.put(b"b", ".html").relative_path


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    RawStore(tmp_path).put(b"x", ".html")
    assert not [p for p in tmp_path.rglob(".tmp-*")]
