from __future__ import annotations

import json
from pathlib import Path


CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "qa_notice_51423_validation.json"
)


def test_notice_51423_validation_corpus_is_well_formed():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    documents = corpus["documents"]

    assert len(documents) == 1
    assert documents[0]["file"] == "notice_51423#05.pdf"

    questions = documents[0]["questions"]
    identifiers = [question["id"] for question in questions]
    assert identifiers == [
        f"notice-{index:03d}"
        for index in range(1, len(questions) + 1)
    ]
    assert len(identifiers) == len(set(identifiers))
    assert sum(question["answerable"] for question in questions) == 18
    assert sum(not question["answerable"] for question in questions) == 2

    for question in questions:
        assert isinstance(question["question"], str)
        assert question["question"].strip()
        assert isinstance(question["answerable"], bool)
        assert isinstance(question["expected_pages"], list)
        assert all(
            isinstance(page, int) and 1 <= page <= 14
            for page in question["expected_pages"]
        )
        assert "expected_answer" in question
        if question["answerable"]:
            assert question["expected_pages"]
            assert isinstance(question["expected_answer"], str)
            assert question["expected_answer"].strip()
        else:
            assert question["expected_pages"] == []
            assert question["expected_answer"] is None
