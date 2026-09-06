from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            sha256.update(block)

    return sha256.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value in the logical form used for stable hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """Return the SHA-256 fingerprint of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
