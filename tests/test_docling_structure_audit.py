from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

from kaliok.experiments.docling_retrieval import (
    DoclingSourceBlock,
    build_docling_corpus,
    build_local_semantic_docling_corpus,
    build_native_docling_corpus,
)


AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "docling_structure_audit.py"
)
spec = importlib.util.spec_from_file_location(
    "docling_structure_audit_tests",
    AUDIT_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Impossible de charger {AUDIT_PATH}")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def _block(
    block_type: str,
    content: str,
    *,
    order: int,
    page: int = 1,
    self_ref: str | None = None,
    parent_ref: str | None = "#/body",
    children_refs: tuple[str, ...] = (),
) -> DoclingSourceBlock:
    return DoclingSourceBlock(
        block_id=uuid4(),
        page=page,
        block_type=block_type,
        content=content,
        reading_order=order,
        indexable=True,
        content_layer="body",
        self_ref=self_ref,
        parent_ref=parent_ref,
        children_refs=children_refs,
    )


def test_detects_introductory_block_followed_by_sibling_paragraphs():
    blocks = [
        _block(
            "section_header",
            "MISSIONS",
            order=0,
            self_ref="#/texts/0",
        ),
        _block(
            "text",
            "La chambre exerce une triple mission :",
            order=1,
            self_ref="#/texts/1",
        ),
        _block(
            "text",
            "Elle examine la gestion.",
            order=2,
            self_ref="#/texts/2",
        ),
        _block(
            "text",
            "Elle juge les comptes.",
            order=3,
            self_ref="#/texts/3",
        ),
        _block(
            "text",
            "Elle rend des avis.",
            order=4,
            self_ref="#/texts/4",
        ),
        _block(
            "section_header",
            "SECTION SUIVANTE",
            order=5,
            self_ref="#/texts/5",
        ),
    ]

    patterns = audit.detect_structural_patterns("ctc.pdf", blocks)

    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.section == "MISSIONS"
    assert len(pattern.elements) == 3
    assert pattern.classifications == ("PARAGRAPHES FRÈRES",)
    assert pattern.reconstructible_from_relations is False


def test_explicit_list_relations_are_reconstructible_and_v3_complete():
    intro = _block(
        "text",
        "Les objectifs sont :",
        order=1,
        self_ref="#/texts/1",
        parent_ref="#/groups/0",
    )
    first = _block(
        "list_item",
        "Former les équipes",
        order=2,
        self_ref="#/texts/2",
        parent_ref="#/groups/0",
    )
    second = _block(
        "list_item",
        "Partager les méthodes",
        order=3,
        self_ref="#/texts/3",
        parent_ref="#/groups/0",
    )
    parent = _block(
        "list",
        "",
        order=10,
        self_ref="#/groups/0",
        children_refs=(intro.self_ref, first.self_ref, second.self_ref),
    )
    blocks = [intro, first, second, parent]

    pattern = audit.detect_structural_patterns("list.pdf", blocks)[0]
    v1 = build_docling_corpus(blocks)
    v3 = build_native_docling_corpus(blocks)
    v4 = build_local_semantic_docling_corpus(blocks)
    audited = audit.audit_patterns([pattern], v1.units, v3, v4)[0]

    assert pattern.classifications == ("GROUPE", "LISTE")
    assert pattern.reconstructible_from_relations is True
    assert [parent.self_ref for parent in pattern.relation_parents] == [
        "#/groups/0"
    ]
    assert audited.v1.complete is False
    assert audited.v1.fragmented is True
    assert audited.v3.complete is True
    assert audited.v3.fragmented is False


def test_picture_children_are_classified_as_picture_text():
    intro = _block(
        "text",
        "Les étapes sont :",
        order=1,
        self_ref="#/texts/1",
        parent_ref="#/pictures/0",
    )
    first = _block(
        "text",
        "Nous étudions",
        order=2,
        self_ref="#/texts/2",
        parent_ref="#/pictures/0",
    )
    second = _block(
        "text",
        "Nous fabriquons",
        order=3,
        self_ref="#/texts/3",
        parent_ref="#/pictures/0",
    )
    picture = _block(
        "picture",
        "",
        order=10,
        self_ref="#/pictures/0",
        children_refs=(intro.self_ref, first.self_ref, second.self_ref),
    )

    pattern = audit.detect_structural_patterns(
        "picture.pdf",
        [intro, first, second, picture],
    )[0]

    assert pattern.classifications == ("PICTURE_TEXT",)
    assert pattern.reconstructible_from_relations is True


def test_v3_grouping_items_without_intro_is_partial_and_fragmented():
    intro = _block(
        "text",
        "Les recommandations sont :",
        order=1,
        self_ref="#/texts/1",
    )
    first = _block(
        "list_item",
        "Fiabiliser les comptes",
        order=2,
        self_ref="#/texts/2",
        parent_ref="#/groups/0",
    )
    second = _block(
        "list_item",
        "Renforcer le pilotage",
        order=3,
        self_ref="#/texts/3",
        parent_ref="#/groups/0",
    )
    parent = _block(
        "list",
        "",
        order=10,
        self_ref="#/groups/0",
        children_refs=(first.self_ref, second.self_ref),
    )
    blocks = [intro, first, second, parent]

    pattern = audit.detect_structural_patterns("partial.pdf", blocks)[0]
    v1 = build_docling_corpus(blocks)
    v3 = build_native_docling_corpus(blocks)
    v4 = build_local_semantic_docling_corpus(blocks)
    audited = audit.audit_patterns([pattern], v1.units, v3, v4)[0]

    assert pattern.reconstructible_from_relations is False
    assert audited.v3.complete is False
    assert audited.v3.partial is True
    assert audited.v3.fragmented is True


def test_introductory_text_followed_by_table_is_detected():
    blocks = [
        _block(
            "text",
            "Les résultats suivants sont :",
            order=1,
            self_ref="#/texts/1",
        ),
        _block(
            "table",
            "Année | Valeur\n2013 | 15",
            order=2,
            self_ref="#/tables/0",
        ),
    ]

    pattern = audit.detect_structural_patterns("table.pdf", blocks)[0]

    assert pattern.classifications == ("TABLE",)
    assert len(pattern.elements) == 1
    assert pattern.reconstructible_from_relations is False


def test_detection_order_is_deterministic_by_page_and_reading_order():
    blocks = [
        _block(
            "text",
            "Les mesures sont :",
            page=2,
            order=10,
            self_ref="#/texts/10",
        ),
        _block("list_item", "Mesure B", page=2, order=12),
        _block("list_item", "Mesure A", page=2, order=11),
    ]

    pattern = audit.detect_structural_patterns("ordered.pdf", blocks)[0]

    assert [element.content for element in pattern.elements] == [
        "Mesure A",
        "Mesure B",
    ]
