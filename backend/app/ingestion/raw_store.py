"""Content-addressed storage for raw downloads.

Files are stored under their SHA-256, so the same bytes are never written twice,
and every derived artefact (parsed blocks, chunks, embeddings) can be rebuilt from
the exact bytes we fetched. Writes are atomic: a crash never leaves a half file.
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredBlob:
    sha256: str
    relative_path: str
    created: bool  # False if identical bytes were already stored


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, data: bytes, suffix: str) -> StoredBlob:
        sha = hashlib.sha256(data).hexdigest()
        rel = f"{sha[:2]}/{sha}{suffix}"
        path = self.root / rel
        if path.exists():
            return StoredBlob(sha, rel, created=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # atomic on the same filesystem
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return StoredBlob(sha, rel, created=True)

    def get(self, relative_path: str) -> bytes:
        return (self.root / relative_path).read_bytes()
