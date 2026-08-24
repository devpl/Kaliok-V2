from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from kaliok.documents.models import (
    DocumentContent,
    DocumentPage,
    TextBlock,
)


DOCLING_COLLECTIONS = ("texts", "groups", "tables", "pictures")
NON_INDEXABLE_LABELS = {"page_header", "page_footer"}


def document_content_from_docling(
    document: dict[str, Any],
) -> DocumentContent:
    """Convert a serialized DoclingDocument to Kaliok's in-memory model."""
    registry, collection_order = _build_registry(document)
    ordered_refs, inherited_layers = _document_order(document, registry)

    for ref in collection_order:
        if ref not in ordered_refs:
            ordered_refs.append(ref)

    blocks: list[TextBlock] = []

    for reading_order, ref in enumerate(ordered_refs):
        item = registry[ref]
        content_layer = (
            item.get("content_layer")
            or inherited_layers.get(ref)
        )
        provenances = _provenances(item)
        page_no = _page_number(item, provenances, registry)

        if page_no is None:
            continue

        label = str(item.get("label") or "text")
        raw_bbox, coordinate_system = _main_bbox(
            item,
            provenances,
        )
        bbox_x, bbox_y, bbox_width, bbox_height = (
            _bbox_dimensions(raw_bbox)
        )
        parent_ref = _reference(item.get("parent"))
        heading_level = _heading_level(item)
        indexable = not (
            label in NON_INDEXABLE_LABELS
            or content_layer == "furniture"
        )
        text = _item_text(item, label)
        extra_data: dict[str, Any] = {
            "docling_self_ref": ref,
            "docling_parent_ref": parent_ref,
            "content_layer": content_layer,
            "heading_level": heading_level,
            "provenances": provenances,
            "indexable": indexable,
        }

        children = [
            child_ref
            for child_ref in (
                _reference(child)
                for child in item.get("children", [])
            )
            if child_ref is not None
        ]
        if children:
            extra_data["docling_children"] = children

        if label == "table":
            extra_data["table"] = item.get("data", {})

        blocks.append(
            TextBlock(
                text=text,
                page=page_no,
                extraction_method="structured",
                extraction_engine="docling",
                confidence=_confidence(provenances),
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                coordinate_system=coordinate_system,
                block_type=label,
                reading_order=reading_order,
                self_ref=ref,
                parent_ref=parent_ref,
                content_layer=content_layer,
                heading_level=heading_level,
                bbox=raw_bbox,
                provenances=provenances,
                indexable=indexable,
                extra_data=extra_data,
            )
        )

    page_count = _page_count(document, blocks)

    origin = document.get("origin")
    origin_filename = (
        origin.get("filename") if isinstance(origin, dict) else None
    )
    return DocumentContent(
        source=str(
            document.get("name") or origin_filename or "docling-document"
        ),
        page_count=page_count,
        blocks=blocks,
        pages=[
            _document_page(document, page_number)
            for page_number in range(1, page_count + 1)
        ],
    )


def _build_registry(
    document: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    registry: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for collection_name in DOCLING_COLLECTIONS:
        collection = document.get(collection_name, [])
        if not isinstance(collection, list):
            continue

        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                continue
            ref = str(
                item.get("self_ref")
                or f"#/{collection_name}/{index}"
            )
            registry[ref] = item
            order.append(ref)

    return registry, order


def _document_order(
    document: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    ordered: list[str] = []
    inherited_layers: dict[str, str] = {}
    visited: set[str] = set()

    def visit(value: Any, inherited_layer: str | None = None) -> None:
        ref = _reference(value)
        item = registry.get(ref) if ref is not None else value

        if not isinstance(item, dict):
            return

        current_layer = item.get("content_layer") or inherited_layer

        if ref is not None and ref in registry:
            if ref in visited:
                return
            visited.add(ref)
            ordered.append(ref)
            if current_layer is not None:
                inherited_layers[ref] = str(current_layer)

        for child in item.get("children", []):
            visit(child, str(current_layer) if current_layer else None)

    visit(document.get("body"), "body")
    visit(document.get("furniture"), "furniture")

    return ordered, inherited_layers


def _reference(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        ref = value.get("$ref") or value.get("ref") or value.get("self_ref")
        return str(ref) if ref is not None else None
    return None


def _provenances(item: dict[str, Any]) -> list[dict[str, Any]]:
    value = item.get("prov", item.get("provenance", []))
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    return []


def _page_number(
    item: dict[str, Any],
    provenances: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> int | None:
    value = item.get("page_no")
    if value is None and provenances:
        value = provenances[0].get("page_no")
    if value is not None:
        return int(value)

    for child in item.get("children", []):
        child_ref = _reference(child)
        child_item = registry.get(child_ref) if child_ref else None
        if child_item is None:
            continue
        child_page = _page_number(
            child_item,
            _provenances(child_item),
            registry,
        )
        if child_page is not None:
            return child_page
    return None


def _main_bbox(
    item: dict[str, Any],
    provenances: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    bbox = item.get("bbox")
    provenance = provenances[0] if provenances else {}
    if bbox is None:
        bbox = provenance.get("bbox")

    if not isinstance(bbox, dict):
        return None, None

    coordinate_system = (
        bbox.get("coord_origin")
        or provenance.get("coord_origin")
        or item.get("coord_origin")
    )
    return bbox, (
        str(coordinate_system)
        if coordinate_system is not None
        else None
    )


def _bbox_dimensions(
    bbox: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if bbox is None:
        return None, None, None, None

    if all(key in bbox for key in ("l", "t", "r", "b")):
        left = float(bbox["l"])
        top = float(bbox["t"])
        right = float(bbox["r"])
        bottom = float(bbox["b"])
        return (
            min(left, right),
            min(top, bottom),
            abs(right - left),
            abs(bottom - top),
        )

    if all(key in bbox for key in ("x", "y", "width", "height")):
        return (
            float(bbox["x"]),
            float(bbox["y"]),
            float(bbox["width"]),
            float(bbox["height"]),
        )

    return None, None, None, None


def _heading_level(item: dict[str, Any]) -> int | None:
    value = item.get("level", item.get("heading_level"))
    return int(value) if value is not None else None


def _item_text(item: dict[str, Any], label: str) -> str:
    text = item.get("text")
    if text is not None and str(text).strip():
        return str(text)
    if label != "table":
        return ""

    data = item.get("data")
    if not isinstance(data, dict):
        return ""

    cells = data.get("table_cells", [])
    if not isinstance(cells, list):
        return ""

    rows: dict[int, dict[int, str]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = int(cell.get("start_row_offset_idx", 0))
        column = int(cell.get("start_col_offset_idx", 0))
        rows.setdefault(row, {})[column] = str(cell.get("text", ""))

    return "\n".join(
        " | ".join(columns[index] for index in sorted(columns))
        for _, columns in sorted(rows.items())
    )


def _confidence(
    provenances: Iterable[dict[str, Any]],
) -> float | None:
    values = [
        float(value)
        for provenance in provenances
        if (value := provenance.get("confidence")) is not None
    ]
    return sum(values) / len(values) if values else None


def _page_count(
    document: dict[str, Any],
    blocks: list[TextBlock],
) -> int:
    pages = document.get("pages")
    if isinstance(pages, (list, dict)) and pages:
        return len(pages)
    return max((block.page for block in blocks), default=0)


def _document_page(
    document: dict[str, Any],
    page_number: int,
) -> DocumentPage:
    pages = document.get("pages", {})
    raw_page: Any = None
    if isinstance(pages, dict):
        raw_page = pages.get(str(page_number), pages.get(page_number))
    elif isinstance(pages, list) and page_number <= len(pages):
        raw_page = pages[page_number - 1]

    size = raw_page.get("size", {}) if isinstance(raw_page, dict) else {}
    return DocumentPage(
        page=page_number,
        width=size.get("width"),
        height=size.get("height"),
        perception_mode="docling",
    )
