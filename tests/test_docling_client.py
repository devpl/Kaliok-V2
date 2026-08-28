from __future__ import annotations

from pathlib import Path

import pytest
import requests

from kaliok.documents import docling_client
from kaliok.documents.docling_client import (
    DoclingClientError,
    convert_pdf_with_docling,
    convert_pdf_with_docling_async,
)
from kaliok.indexing import docling as docling_indexing


class FakeResponse:
    def __init__(self, payload, *, http_error=None):
        self.payload = payload
        self.http_error = http_error

    def raise_for_status(self):
        if self.http_error is not None:
            raise self.http_error

    def json(self):
        return self.payload


def test_docling_client_returns_json_content_and_sends_expected_multipart(
    monkeypatch,
    tmp_path,
):
    pdf_path = tmp_path / "document-échantillon.pdf"
    pdf_bytes = b"%PDF-1.4\ncontenu"
    pdf_path.write_bytes(pdf_bytes)
    expected = {"name": "Document été", "texts": []}
    received = {}

    def fake_post(url, *, files, data, timeout):
        filename, file_object, content_type = files["files"]
        received.update(
            {
                "url": url,
                "filename": filename,
                "content": file_object.read(),
                "content_type": content_type,
                "data": data,
                "timeout": timeout,
            }
        )
        return FakeResponse({"document": {"json_content": expected}})

    monkeypatch.setenv("KALIOK_DOCLING_URL", "http://docling.test/")
    monkeypatch.setattr(docling_client.requests, "post", fake_post)

    result = convert_pdf_with_docling(pdf_path, timeout=17)

    assert result is expected
    assert received == {
        "url": "http://docling.test/v1/convert/file",
        "filename": "document-échantillon.pdf",
        "content": pdf_bytes,
        "content_type": "application/pdf",
        "data": {
            "to_formats": "json",
            "image_export_mode": "placeholder",
            "do_ocr": "true",
            "ocr_engine": "auto",
            "table_mode": "accurate",
        },
        "timeout": 17,
    }


def test_docling_client_reports_http_error(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    response = FakeResponse(
        {},
        http_error=requests.HTTPError("500 Server Error"),
    )
    monkeypatch.setattr(
        docling_client.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(DoclingClientError, match="Échec HTTP") as error:
        convert_pdf_with_docling(
            pdf_path,
            base_url="http://docling.test",
        )

    assert isinstance(error.value.__cause__, requests.HTTPError)


def test_docling_client_reports_timeout(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")

    def timeout(*args, **kwargs):
        raise requests.Timeout("conversion trop longue")

    monkeypatch.setattr(docling_client.requests, "post", timeout)

    with pytest.raises(DoclingClientError, match="Délai dépassé") as error:
        convert_pdf_with_docling(
            pdf_path,
            base_url="http://docling.test",
        )

    assert isinstance(error.value.__cause__, requests.Timeout)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"document": {}},
        {"document": {"json_content": None}},
        {"document": {"json_content": "{}"}},
    ],
)
def test_docling_client_rejects_missing_json_content(
    monkeypatch,
    tmp_path,
    payload,
):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        docling_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(
        DoclingClientError,
        match=r"document\.json_content",
    ):
        convert_pdf_with_docling(
            pdf_path,
            base_url="http://docling.test",
        )


def test_store_pdf_with_docling_uses_json_version(monkeypatch):
    session = object()
    version = object()
    run = object()
    json_content = {
        "name": "document",
        "version": "1.10.0",
        "texts": [],
    }
    received = {}

    monkeypatch.setattr(
        docling_indexing,
        "convert_pdf_with_docling",
        lambda *args, **kwargs: json_content,
    )

    def fake_store(received_session, received_version, document, **kwargs):
        received.update(kwargs)
        return run

    monkeypatch.setattr(
        docling_indexing,
        "store_docling_document",
        fake_store,
    )

    result = docling_indexing.store_pdf_with_docling(
        session,
        version,
        Path("document.pdf"),
    )

    assert result is run
    assert received == {"engine_version": "1.10.0"}


def test_store_pdf_with_docling_explicit_version_has_priority(monkeypatch):
    session = object()
    version = object()
    run = object()
    json_content = {
        "name": "document",
        "version": "1.10.0",
        "texts": [],
    }
    calls = []

    def fake_convert(path, *, base_url, timeout):
        calls.append(("convert", path, base_url, timeout))
        return json_content

    def fake_store(received_session, received_version, document, **kwargs):
        calls.append(
            (
                "store",
                received_session,
                received_version,
                document,
                kwargs,
            )
        )
        return run

    monkeypatch.setattr(
        docling_indexing,
        "convert_pdf_with_docling",
        fake_convert,
    )
    monkeypatch.setattr(
        docling_indexing,
        "store_docling_document",
        fake_store,
    )

    result = docling_indexing.store_pdf_with_docling(
        session,
        version,
        Path("document.pdf"),
        base_url="http://docling.test",
        timeout=12,
        engine_version="1.2.3",
    )

    assert result is run
    assert calls == [
        (
            "convert",
            Path("document.pdf"),
            "http://docling.test",
            12,
        ),
        (
            "store",
            session,
            version,
            json_content,
            {"engine_version": "1.2.3"},
        ),
    ]


def test_docling_async_submits_polls_and_recovers_result(
    monkeypatch,
    tmp_path,
):
    pdf_path = tmp_path / "document-échantillon.pdf"
    pdf_bytes = b"%PDF async"
    pdf_path.write_bytes(pdf_bytes)
    expected = {"version": "2.1.0", "texts": [{"text": "été"}]}
    statuses = iter(("pending", "started", "success"))
    calls = []
    sleeps = []

    def fake_post(url, *, files, data, timeout):
        filename, file_object, content_type = files["files"]
        calls.append(
            (
                "POST",
                url,
                filename,
                file_object.read(),
                content_type,
                data,
                timeout,
            )
        )
        return FakeResponse({"task_id": "task-123"})

    def fake_get(url, *, timeout):
        calls.append(("GET", url, timeout))
        if "/status/poll/" in url:
            return FakeResponse({"task_status": next(statuses)})
        return FakeResponse({"document": {"json_content": expected}})

    monkeypatch.setattr(docling_client.requests, "post", fake_post)
    monkeypatch.setattr(docling_client.requests, "get", fake_get)
    monkeypatch.setattr(docling_client.time, "sleep", sleeps.append)

    result = convert_pdf_with_docling_async(
        pdf_path,
        base_url="http://docling.test/",
        submit_timeout=17,
        poll_interval=0.25,
        overall_timeout=60,
    )

    assert result is expected
    assert calls[0] == (
        "POST",
        "http://docling.test/v1/convert/file/async",
        "document-échantillon.pdf",
        pdf_bytes,
        "application/pdf",
        {
            "to_formats": "json",
            "image_export_mode": "placeholder",
            "do_ocr": "true",
            "ocr_engine": "auto",
            "table_mode": "accurate",
        },
        17,
    )
    assert [call[1] for call in calls[1:]] == [
        "http://docling.test/v1/status/poll/task-123",
        "http://docling.test/v1/status/poll/task-123",
        "http://docling.test/v1/status/poll/task-123",
        "http://docling.test/v1/result/task-123",
    ]
    assert sleeps == [0.25, 0.25]


def test_docling_async_reports_task_failure(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    monkeypatch.setattr(
        docling_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"task_id": "failed-task"}),
    )
    monkeypatch.setattr(
        docling_client.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            {"task_status": "failure", "error": "OCR impossible"}
        ),
    )

    with pytest.raises(
        DoclingClientError,
        match="failed-task.*échoué.*OCR impossible",
    ):
        convert_pdf_with_docling_async(
            pdf_path,
            base_url="http://docling.test",
        )


def test_docling_async_enforces_overall_timeout(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    clock = {"now": 0.0}
    poll_calls = []

    monkeypatch.setattr(
        docling_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"task_id": "slow-task"}),
    )

    def fake_get(*args, **kwargs):
        poll_calls.append(args[0])
        return FakeResponse({"task_status": "pending"})

    monkeypatch.setattr(docling_client.requests, "get", fake_get)
    monkeypatch.setattr(
        docling_client.time,
        "perf_counter",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        docling_client.time,
        "sleep",
        lambda duration: clock.__setitem__("now", clock["now"] + duration),
    )

    with pytest.raises(DoclingClientError, match="Délai global dépassé"):
        convert_pdf_with_docling_async(
            pdf_path,
            base_url="http://docling.test",
            poll_interval=2,
            overall_timeout=3,
        )

    assert len(poll_calls) == 2


@pytest.mark.parametrize("failing_stage", ["submit", "poll", "result"])
def test_docling_async_reports_http_errors(
    monkeypatch,
    tmp_path,
    failing_stage,
):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    http_error = requests.HTTPError("404 Not Found")
    post_response = FakeResponse(
        {"task_id": "task-404"},
        http_error=http_error if failing_stage == "submit" else None,
    )
    monkeypatch.setattr(
        docling_client.requests,
        "post",
        lambda *args, **kwargs: post_response,
    )

    def fake_get(url, **kwargs):
        if failing_stage == "poll" and "/status/poll/" in url:
            return FakeResponse({}, http_error=http_error)
        if "/status/poll/" in url:
            return FakeResponse({"task_status": "success"})
        return FakeResponse(
            {"document": {"json_content": {}}},
            http_error=http_error if failing_stage == "result" else None,
        )

    monkeypatch.setattr(docling_client.requests, "get", fake_get)

    with pytest.raises(DoclingClientError, match="Échec HTTP") as error:
        convert_pdf_with_docling_async(
            pdf_path,
            base_url="http://docling.test",
        )

    assert error.value.__cause__ is http_error


def test_store_pdf_with_docling_async_orchestrates_conversion_and_storage(
    monkeypatch,
):
    session = object()
    version = object()
    run = object()
    document = {"version": "2.2.0", "texts": []}
    calls = []

    def fake_convert(path, **kwargs):
        calls.append(("convert", path, kwargs))
        return document

    def fake_store(received_session, received_version, received, **kwargs):
        calls.append(
            (
                "store",
                received_session,
                received_version,
                received,
                kwargs,
            )
        )
        return run

    monkeypatch.setattr(
        docling_indexing,
        "convert_pdf_with_docling_async",
        fake_convert,
    )
    monkeypatch.setattr(
        docling_indexing,
        "store_docling_document",
        fake_store,
    )

    result = docling_indexing.store_pdf_with_docling_async(
        session,
        version,
        Path("document.pdf"),
        base_url="http://docling.test",
        submit_timeout=12,
        poll_interval=0.5,
        overall_timeout=90,
    )

    assert result is run
    assert calls == [
        (
            "convert",
            Path("document.pdf"),
            {
                "base_url": "http://docling.test",
                "submit_timeout": 12,
                "poll_interval": 0.5,
                "overall_timeout": 90,
            },
        ),
        (
            "store",
            session,
            version,
            document,
            {"engine_version": "2.2.0"},
        ),
    ]
