from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from kaliok.api.dependencies import get_session
from kaliok.api.main import app


class FakeSession:
    def __init__(self):
        self.commit_count = 0
        self.rollback_count = 0

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


class FakeRuntime:
    def __init__(
        self,
        *,
        document_id,
        document_version_id,
        indexed,
        answer=None,
        error=None,
    ):
        self.document_id = document_id
        self.document_version_id = document_version_id
        self._indexed = indexed
        self._answer = answer
        self._error = error

        self.index_count = 0
        self.question = None

    def is_indexed(self):
        return self._indexed

    def index(self):
        self.index_count += 1
        return 2

    def answer(self, question):
        self.question = question

        if self._error is not None:
            raise self._error

        return self._answer


def make_answer(
    *,
    document_id,
    document_version_id,
):
    source_1_id = uuid4()
    source_2_id = uuid4()

    def ranked(
        *,
        rank,
        score,
        source_id,
        source_unit_id,
        text,
    ):
        provenance = SimpleNamespace(
            document_id=document_id,
            document_version_id=document_version_id,
            metadata={
                "normalized_content_unit_id": source_id,
                "source_unit_id": source_unit_id,
            },
        )

        unit = SimpleNamespace(
            text=text,
            provenance=provenance,
        )

        return SimpleNamespace(
            rank=rank,
            score=score,
            unit=unit,
        )

    candidates = (
        ranked(
            rank=1,
            score=0.91,
            source_id=source_1_id,
            source_unit_id="paragraph-2",
            text="Premier passage pertinent.",
        ),
        ranked(
            rank=2,
            score=0.82,
            source_id=source_2_id,
            source_unit_id="paragraph-4",
            text="Deuxième passage pertinent.",
        ),
    )

    return SimpleNamespace(
        text="Réponse produite par le RAG.",
        metadata={
            "model": "generation-test",
        },
        context=SimpleNamespace(
            candidates=candidates,
        ),
    )


def test_rag_answer_reuses_existing_index(monkeypatch):
    from kaliok.api import rag as rag_api

    session = FakeSession()
    document_id = uuid4()
    version_id = uuid4()

    answer = make_answer(
        document_id=document_id,
        document_version_id=version_id,
    )

    runtime = FakeRuntime(
        document_id=document_id,
        document_version_id=version_id,
        indexed=True,
        answer=answer,
    )

    captured = {}

    def fake_factory(
            current_session,
            *,
            reference,
    ):
        captured["session"] = current_session
        captured["reference"] = reference
        return runtime

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        rag_api,
        "create_normalized_rag_runtime",
        fake_factory,
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/rag/answer",
            json={
                "document_id": str(document_id),
                "question": "Quelle est la réponse ?",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    payload = response.json()

    assert payload["document_id"] == str(document_id)
    assert payload["document_version_id"] == str(version_id)
    assert payload["answer"] == "Réponse produite par le RAG."
    assert payload["generation_model"] == "generation-test"
    assert payload["indexed_now"] is False

    assert len(payload["sources"]) == 2

    assert payload["sources"][0] == {
        "rank": 1,
        "score": 0.91,
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "normalized_content_unit_id": str(
            answer.context.candidates[
                0
            ].unit.provenance.metadata[
                "normalized_content_unit_id"
            ]
        ),
        "source_unit_id": "paragraph-2",
        "text": "Premier passage pertinent.",
    }

    assert captured["session"] is session
    assert captured["reference"].document_id == document_id
    assert captured["reference"].document_version_id is None


    assert runtime.index_count == 0
    assert runtime.question == "Quelle est la réponse ?"

    assert session.commit_count == 0
    assert session.rollback_count == 0


def test_rag_answer_indexes_missing_version(monkeypatch):
    from kaliok.api import rag as rag_api

    session = FakeSession()
    document_id = uuid4()
    version_id = uuid4()

    runtime = FakeRuntime(
        document_id=document_id,
        document_version_id=version_id,
        indexed=False,
        answer=make_answer(
            document_id=document_id,
            document_version_id=version_id,
        ),
    )

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        rag_api,
        "create_normalized_rag_runtime",
        lambda current_session, **kwargs: runtime,
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/rag/answer",
            json={
                "document_id": str(document_id),
                "question": "Question de test",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    payload = response.json()

    assert payload["indexed_now"] is True

    assert runtime.index_count == 1
    assert runtime.question == "Question de test"

    assert session.commit_count == 1
    assert session.rollback_count == 0


def test_rag_answer_returns_400_for_expected_error(monkeypatch):
    from kaliok.api import rag as rag_api

    session = FakeSession()
    document_id = uuid4()

    def fail_factory(current_session, **kwargs):
        raise ValueError(
            "Le document ne possède pas de contenu normalisé."
        )

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        rag_api,
        "create_normalized_rag_runtime",
        fail_factory,
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/rag/answer",
            json={
                "document_id": str(document_id),
                "question": "Question",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Le document ne possède pas de contenu normalisé.",
    }

    assert session.commit_count == 0
    assert session.rollback_count == 1


def test_rag_answer_rolls_back_unexpected_error(monkeypatch):
    from kaliok.api import rag as rag_api

    session = FakeSession()
    document_id = uuid4()
    version_id = uuid4()

    runtime = FakeRuntime(
        document_id=document_id,
        document_version_id=version_id,
        indexed=True,
        error=RuntimeError("Erreur RAG inattendue"),
    )

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        rag_api,
        "create_normalized_rag_runtime",
        lambda current_session, **kwargs: runtime,
    )

    try:
        client = TestClient(
            app,
            raise_server_exceptions=False,
        )

        response = client.post(
            "/rag/answer",
            json={
                "document_id": str(document_id),
                "question": "Question",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500

    assert session.commit_count == 0
    assert session.rollback_count == 1
