from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from kaliok.hashing import calculate_sha256
from kaliok.ingestion.types import (
    DetectedSource,
    IngestionRequest,
    NormalizedContentUnit,
    NormalizedDocument,
)


PLAIN_TEXT_SOURCE_TYPE = "plain_text"
PLAIN_TEXT_MEDIA_TYPE = "text/plain"


class TxtSourceLocationError(ValueError):
    pass


class TxtDecodingError(UnicodeError):
    pass


class TxtSourceIngestor:
    def supports(self, source: DetectedSource) -> bool:
        media_type = source.media_type or source.source.media_type
        return (
            source.source_type == PLAIN_TEXT_SOURCE_TYPE
            and media_type == PLAIN_TEXT_MEDIA_TYPE
        )

    def ingest(
        self,
        request: IngestionRequest,
        source: DetectedSource,
    ) -> NormalizedDocument:
        if not self.supports(source):
            raise ValueError("La source détectée n'est pas un texte simple.")

        path = self._local_path(request.source.uri)
        try:
            raw_content = path.read_bytes()
        except OSError as error:
            raise TxtSourceLocationError(
                f"Impossible de lire la source texte locale : {path}."
            ) from error

        try:
            text = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise TxtDecodingError(
                "La source texte n'est pas encodée en UTF-8 valide."
            ) from error

        units = tuple(
            NormalizedContentUnit(
                order=order,
                content_type="paragraph",
                content=paragraph,
                source_reference=request.source.uri,
                source_unit_id=f"paragraph-{order}",
            )
            for order, paragraph in enumerate(self._paragraphs(text))
        )

        return NormalizedDocument(
            source=source,
            units=units,
            filename=path.name,
            storage_uri=path.resolve().as_uri(),
            content_hash=calculate_sha256(path),
            file_size=len(raw_content),
            mime_type=PLAIN_TEXT_MEDIA_TYPE,
            title=path.stem,
            page_count=None,
            document_family=None,
            document_type="text",
            document_subtype=None,
        )

    @staticmethod
    def _paragraphs(text: str) -> tuple[str, ...]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return tuple(
            block.strip()
            for block in re.split(r"\n[ \t]*\n+", normalized)
            if block.strip()
        )

    @staticmethod
    def _local_path(uri: str | None) -> Path:
        if uri is None or not uri.strip():
            raise TxtSourceLocationError(
                "SourceReference.uri est requis pour une source texte locale."
            )

        parsed = urlparse(uri)
        is_windows_path = (
            len(parsed.scheme) == 1
            and len(uri) >= 3
            and uri[1] == ":"
            and uri[2] in {"/", "\\"}
        )
        if not parsed.scheme or is_windows_path:
            path = Path(uri)
        elif parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise TxtSourceLocationError(
                    "Seules les URI file locales sont prises en charge."
                )
            decoded_path = url2pathname(unquote(parsed.path))
            if os.name == "nt" and re.match(r"^[/\\][A-Za-z]:", decoded_path):
                decoded_path = decoded_path[1:]
            path = Path(decoded_path)
        else:
            raise TxtSourceLocationError(
                "Seuls un chemin local ou une URI file locale sont pris en charge."
            )

        if not path.is_file():
            raise TxtSourceLocationError(
                f"Source texte locale introuvable : {path}."
            )
        return path
