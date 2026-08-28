from __future__ import annotations

import json
from pathlib import Path


CORPUS_PATH = (
    Path(__file__).resolve().parents[1] / "qa_ctc_2013_validation.json"
)


def test_ctc_validation_corpus_has_valid_questions_and_pages():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    documents = corpus["documents"]

    assert len(documents) == 1
    assert documents[0]["file"] == (
        "rapport-d-activit--s-2013-de-la-ctc-NC_1.pdf"
    )

    questions = documents[0]["questions"]
    identifiers = [question["id"] for question in questions]
    assert len(identifiers) == len(set(identifiers))
    assert identifiers == [
        f"ctc-{index:03d}" for index in range(1, len(questions) + 1)
    ]

    for question in questions:
        assert isinstance(question["question"], str)
        assert question["question"].strip()
        assert isinstance(question["answerable"], bool)
        assert isinstance(question["expected_pages"], list)
        assert all(
            isinstance(page, int) and 1 <= page <= 28
            for page in question["expected_pages"]
        )
        if question["answerable"]:
            assert question["expected_pages"]
        else:
            assert question["expected_pages"] == []
