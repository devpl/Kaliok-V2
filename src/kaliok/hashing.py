from __future__ import annotations

import hashlib
from pathlib import Path


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            sha256.update(block)

    return sha256.hexdigest()
