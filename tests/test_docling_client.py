from __future__ import annotations

from pathlib import Path

import pytest
import requests

from kaliok.documents import docling_client
from kaliok.documents.docling_client import (
    DoclingClientError,
    convert_pdf_with_docling,
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


def test_store_pdf_with_docling_is_explicit_orchestration(monkeypatch):
    session = object()
    version = object()
    run = object()
    json_content = {"name": "document", "texts": []}
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
