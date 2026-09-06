from __future__ import annotations

from kaliok.discovery import LexicalCandidateDetector, LexicalTerm
from kaliok.storage.models import NormalizedContentUnit


def _unit(content: str) -> NormalizedContentUnit:
    return NormalizedContentUnit(
        document_version_id="00000000-0000-0000-0000-000000000001",
        unit_index=0,
        content_type="paragraph",
        content=content,
    )


def test_unicode_word_boundaries_avoid_longer_words_and_keep_exact_offsets():
    text = "Nouméa, nouméa — Grand-Nouméa; Nouméaville"
    detector = LexicalCandidateDetector(
        [LexicalTerm("Nouméa", "place_name")],
        case_sensitive=False,
        boundary_policy="unicode_word",
    )

    occurrences = detector.detect(_unit(text))

    assert [item.exact_text for item in occurrences] == ["Nouméa", "nouméa", "Nouméa"]
    assert [(item.start_offset, item.end_offset) for item in occurrences] == [
        (0, 6), (8, 14), (23, 29)
    ]
    assert detector.configuration["boundary_policy"] == "unicode_word"


def test_substring_policy_is_explicit_and_multiword_terms_still_work():
    substring = LexicalCandidateDetector(
        [LexicalTerm("Durand", "person_name")],
        boundary_policy="substring",
    )
    phrase = LexicalCandidateDetector(
        [LexicalTerm("Chambre territoriale des comptes", "organization")],
        boundary_policy="unicode_word",
    )

    assert len(substring.detect(_unit("Durandville"))) == 1
    match = phrase.detect(_unit("La Chambre territoriale des comptes siège ici."))[0]
    assert match.exact_text == "Chambre territoriale des comptes"
    assert match.start_offset == 3
