from __future__ import annotations

from uuid import uuid4

from django.test import Client, override_settings
from django.urls import reverse

from kaliok.ui.core_ui import views


TEST_API_BASE_URL = "http://kaliok-api-test.invalid"


def make_document(document_id):
    version_id = uuid4()

    return {
        "id": str(document_id),
        "title": "Document de test",
        "status": "active",
        "document_family": None,
        "language": "fr",
        "created_at": "2026-09-01T10:00:00Z",
        "current_version": {
            "id": str(version_id),
            "version_number": 1,
            "filename": "document-test.txt",
            "mime_type": "text/plain",
            "file_size": 123,
            "page_count": None,
            "processing_status": "pending",
            "readability_status": "unknown",
            "storage_uri": "file:///document-test.txt",
        },
        "versions": [],
    }


def make_client():
    return Client(
        HTTP_HOST="localhost",
    )


def test_document_detail_get_displays_rag_form(monkeypatch):
    document_id = uuid4()

    monkeypatch.setattr(
        views,
        "get_api_document",
        lambda current_document_id: make_document(
            current_document_id
        ),
    )

    client = make_client()

    response = client.get(
        reverse(
            "document_detail",
            kwargs={"document_id": document_id},
        )
    )

    assert response.status_code == 200

    content = response.content.decode()

    assert "Interroger le document" in content
    assert "Poser la question" in content
    assert 'name="question"' in content

    assert "<h3>Réponse</h3>" not in content
    assert "Sources utilisées" not in content


def test_document_detail_post_calls_rag_api(monkeypatch):
    document_id = uuid4()
    source_id = uuid4()
    version_id = uuid4()

    monkeypatch.setattr(
        views,
        "get_api_document",
        lambda current_document_id: make_document(
            current_document_id
        ),
    )

    captured = {}

    def fake_ask_api_rag(
        *,
        document_id,
        question,
    ):
        captured["document_id"] = document_id
        captured["question"] = question

        return {
            "document_id": str(document_id),
            "document_version_id": str(version_id),
            "answer": "Le deuxième paragraphe contient la réponse.",
            "generation_model": "generation-test",
            "indexed_now": False,
            "sources": [
                {
                    "rank": 1,
                    "score": 0.91,
                    "document_id": str(document_id),
                    "document_version_id": str(version_id),
                    "normalized_content_unit_id": str(source_id),
                    "source_unit_id": "paragraph-1",
                    "text": "Deuxième paragraphe pertinent.",
                }
            ],
        }

    monkeypatch.setattr(
        views,
        "ask_api_rag",
        fake_ask_api_rag,
    )

    client = make_client()

    response = client.post(
        reverse(
            "document_detail",
            kwargs={"document_id": document_id},
        ),
        {
            "question": "Que contient le deuxième paragraphe ?",
        },
    )

    assert response.status_code == 200

    assert captured == {
        "document_id": document_id,
        "question": "Que contient le deuxième paragraphe ?",
    }

    content = response.content.decode()

    assert "Le deuxième paragraphe contient la réponse." in content
    assert "Sources utilisées" in content
    assert "Deuxième paragraphe pertinent." in content
    assert str(source_id) in content
    assert "0,910" in content or "0.910" in content


def test_document_detail_invalid_question_does_not_call_rag(
    monkeypatch,
):
    document_id = uuid4()

    monkeypatch.setattr(
        views,
        "get_api_document",
        lambda current_document_id: make_document(
            current_document_id
        ),
    )

    rag_called = False

    def fake_ask_api_rag(**kwargs):
        nonlocal rag_called
        rag_called = True
        raise AssertionError(
            "Le RAG ne doit pas être appelé."
        )

    monkeypatch.setattr(
        views,
        "ask_api_rag",
        fake_ask_api_rag,
    )

    client = make_client()

    response = client.post(
        reverse(
            "document_detail",
            kwargs={"document_id": document_id},
        ),
        {
            "question": "",
        },
    )

    assert response.status_code == 200
    assert rag_called is False

    content = response.content.decode()

    assert "Ce champ est obligatoire." in content


def test_document_detail_displays_rag_error(monkeypatch):
    document_id = uuid4()

    monkeypatch.setattr(
        views,
        "get_api_document",
        lambda current_document_id: make_document(
            current_document_id
        ),
    )

    monkeypatch.setattr(
        views,
        "ask_api_rag",
        lambda **kwargs: None,
    )

    client = make_client()

    response = client.post(
        reverse(
            "document_detail",
            kwargs={"document_id": document_id},
        ),
        {
            "question": "Question de test",
        },
    )

    assert response.status_code == 200

    content = response.content.decode()

    assert "Le service RAG n" in content
    assert "pas pu répondre à la question." in content


@override_settings(
    KALIOK_API_BASE_URL=TEST_API_BASE_URL,
)
def test_ask_api_rag_calls_fastapi(monkeypatch):
    document_id = uuid4()

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "document_id": str(document_id),
                "document_version_id": str(uuid4()),
                "answer": "Réponse de test",
                "generation_model": "generation-test",
                "indexed_now": False,
                "sources": [],
            }

    def fake_post(
        url,
        *,
        json,
        timeout,
    ):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        views.httpx,
        "post",
        fake_post,
    )

    result = views.ask_api_rag(
        document_id=document_id,
        question="Ma question",
    )

    assert result is not None
    assert result["answer"] == "Réponse de test"

    assert captured == {
        "url": (
            f"{TEST_API_BASE_URL}"
            "/rag/answer"
        ),
        "json": {
            "document_id": str(document_id),
            "question": "Ma question",
            "top_k": 5,
        },
        "timeout": 300.0,
    }