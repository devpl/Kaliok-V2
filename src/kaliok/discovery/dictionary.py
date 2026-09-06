from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaliok.discovery.detectors import LexicalCandidateDetector, LexicalTerm
from kaliok.hashing import canonical_json_hash


@dataclass(frozen=True)
class LexicalDictionaryMetadata:
    dictionary_key: str
    version: str
    name: str
    description: str | None = None
    language: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dictionary_key": self.dictionary_key,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "language": self.language,
        }


@dataclass(frozen=True)
class LoadedLexicalDictionary:
    metadata: LexicalDictionaryMetadata
    terms: tuple[LexicalTerm, ...]
    dictionary_hash: str
    case_sensitive: bool = False
    boundary_policy: str = "unicode_word"

    def __iter__(self):
        """Préserve le déballage historique ``detector, configuration``."""
        yield self.detector()
        yield self.snapshot()

    def detector(self) -> LexicalCandidateDetector:
        return LexicalCandidateDetector(
            self.terms,
            case_sensitive=self.case_sensitive,
            boundary_policy=self.boundary_policy,
            dictionary_identity={**self.metadata.as_dict(), "hash": self.dictionary_hash},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.metadata.as_dict(),
            "hash": self.dictionary_hash,
            "case_sensitive": self.case_sensitive,
            "boundary_policy": self.boundary_policy,
            "terms": [_term_payload(term) for term in self.terms],
        }


def _term_payload(term: LexicalTerm) -> dict[str, Any]:
    return {
        "value": term.value,
        "candidate_type": term.candidate_type,
        "normalized_value": term.normalized_value,
        "confidence": term.confidence,
        "payload": term.payload,
    }


def load_lexical_dictionary(path: str | Path) -> LoadedLexicalDictionary:
    dictionary_path = Path(path)
    try:
        value = json.loads(dictionary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Impossible de charger le dictionnaire lexical : {dictionary_path}.") from error
    if not isinstance(value, dict):
        raise ValueError("Le dictionnaire lexical doit être un objet JSON.")

    def required_text(key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Le champ '{key}' du dictionnaire est requis et ne peut pas être vide.")
        return item.strip()

    metadata = LexicalDictionaryMetadata(
        dictionary_key=required_text("dictionary_key"),
        version=required_text("version"),
        name=required_text("name"),
        description=_optional_text(value, "description"),
        language=_optional_text(value, "language"),
    )
    raw_terms = value.get("terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("Le dictionnaire lexical doit contenir une liste 'terms' non vide.")
    terms: list[LexicalTerm] = []
    for index, item in enumerate(raw_terms):
        if not isinstance(item, dict):
            raise ValueError(f"Le terme lexical {index} doit être un objet JSON.")
        term_value = item.get("value")
        candidate_type = item.get("candidate_type")
        if not isinstance(term_value, str) or not term_value.strip():
            raise ValueError(f"Le terme lexical {index} doit contenir une 'value' non vide.")
        if not isinstance(candidate_type, str) or not candidate_type.strip():
            raise ValueError(f"Le terme lexical {index} doit contenir un 'candidate_type' non vide.")
        normalized = item.get("normalized_value")
        if normalized is not None and (not isinstance(normalized, str) or not normalized.strip()):
            raise ValueError(f"normalized_value du terme lexical {index} doit être une chaîne non vide.")
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError(f"Le payload du terme lexical {index} doit être un objet.")
        confidence = item.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
            raise ValueError(f"La confiance du terme lexical {index} doit être numérique.")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError(f"La confiance du terme lexical {index} doit être comprise entre 0 et 1.")
        terms.append(LexicalTerm(term_value, candidate_type, normalized, payload, float(confidence) if confidence is not None else None))
    boundary_policy = value.get("boundary_policy", "unicode_word")
    case_sensitive = value.get("case_sensitive", False)
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive doit être un booléen.")
    if boundary_policy not in {"unicode_word", "substring"}:
        raise ValueError("boundary_policy doit être 'unicode_word' ou 'substring'.")
    logical = {
        **metadata.as_dict(),
        "case_sensitive": case_sensitive,
        "boundary_policy": boundary_policy,
        "terms": [_term_payload(term) for term in terms],
    }
    return LoadedLexicalDictionary(
        metadata,
        tuple(terms),
        canonical_json_hash(logical),
        case_sensitive,
        boundary_policy,
    )


def _optional_text(value: dict[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Le champ optionnel '{key}' doit être une chaîne non vide.")
    return item.strip()
