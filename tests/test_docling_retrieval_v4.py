from __future__ import annotations

from uuid import uuid4

import pytest

from kaliok.experiments.docling_retrieval import (
    DoclingSourceBlock,
    build_docling_corpus,
    build_local_semantic_docling_corpus,
    build_native_docling_corpus,
    build_structural_docling_corpus,
)


def _block(
    block_type: str,
    content: str,
    *,
    page: int = 1,
    order: int,
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


def _composites(blocks: list[DoclingSourceBlock]):
    return [
        unit
        for unit in build_local_semantic_docling_corpus(blocks).units
        if unit.is_composite
    ]


def test_v4_combines_adjacent_intro_and_list_and_keeps_v3_atomics():
    heading = _block("section_header", "Section", order=0)
    intro = _block("text", "Les points locaux :", order=1)
    group = _block(
        "list",
        "",
        order=2,
        self_ref="#/groups/list",
        children_refs=("#/texts/a", "#/texts/b"),
    )
    first = _block(
        "list_item",
        "Premier élément",
        order=3,
        self_ref="#/texts/a",
        parent_ref="#/groups/list",
    )
    second = _block(
        "list_item",
        "Second élément",
        order=4,
        self_ref="#/texts/b",
        parent_ref="#/groups/list",
    )
    blocks = [heading, intro, group, first, second]
    v3 = build_native_docling_corpus(blocks)
    v4 = build_local_semantic_docling_corpus(blocks)
    composites = [unit for unit in v4.units if unit.is_composite]

    assert len(composites) == 1
    composite = composites[0]
    assert composite.logical_type == "semantic_group_list"
    assert composite.section_header == "Section"
    assert composite.group_ref == "#/groups/list"
    assert composite.source_block_ids == (
        intro.block_id,
        first.block_id,
        second.block_id,
    )
    assert composite.source_block_types == ("text", "list_item", "list_item")
    assert composite.text == (
        "Section\nLes points locaux :\nPremier élément\nSecond élément"
    )
    atomics = [unit for unit in v4.units if not unit.is_composite]
    assert [unit.text for unit in atomics] == [unit.text for unit in v3.units]
    assert v4.atomic_count == len(v3.units)


def test_v4_combines_bounded_contiguous_text_siblings():
    blocks = [
        _block("section_header", "Section", order=0),
        _block("text", "Introduction locale :", order=1),
        _block("text", "Premier développement", order=2),
        _block("text", "Deuxième développement", order=3),
        _block("text", "Troisième développement", order=4),
    ]

    composites = _composites(blocks)

    assert len(composites) == 1
    assert composites[0].logical_type == "semantic_group_siblings"
    assert len(composites[0].source_block_ids) == 4
    assert composites[0].parent_refs == ("#/body",)
    assert composites[0].text.endswith(
        "Introduction locale :\nPremier développement\n"
        "Deuxième développement\nTroisième développement"
    )


@pytest.mark.parametrize(
    "blocks",
    [
        # Changement de section.
        [
            _block("text", "Intro", order=0),
            _block("text", "Un", order=1),
            _block("section_header", "Autre", order=2),
            _block("text", "Deux", order=3),
        ],
        # Changement de page.
        [
            _block("text", "Intro", page=1, order=0),
            _block("text", "Un", page=1, order=1),
            _block("text", "Deux", page=2, order=2),
        ],
        # Changement de parent.
        [
            _block("text", "Intro", order=0, parent_ref="#/a"),
            _block("text", "Un", order=1, parent_ref="#/a"),
            _block("text", "Deux", order=2, parent_ref="#/b"),
        ],
        # Table intercalée : blocs non adjacents.
        [
            _block("text", "Intro", order=0),
            _block("text", "Un", order=1),
            _block("table", "A | B", order=2),
            _block("text", "Deux", order=3),
        ],
        # Texte d'image exclu des paragraphes frères.
        [
            _block("text", "Intro", order=0, parent_ref="#/pictures/1"),
            _block("text", "Un", order=1, parent_ref="#/pictures/1"),
            _block("text", "Deux", order=2, parent_ref="#/pictures/1"),
        ],
    ],
)
def test_v4_does_not_cross_structural_boundaries(blocks):
    assert _composites(blocks) == []


def test_v4_does_not_turn_long_ordinary_paragraphs_into_a_huge_block():
    blocks = [
        _block("text", f"Paragraphe {index} " + "x" * 260, order=index)
        for index in range(6)
    ]

    corpus = build_local_semantic_docling_corpus(blocks)

    assert corpus.composite_count == 0
    assert corpus.atomic_count == len(build_native_docling_corpus(blocks).units)


def test_v4_refuses_a_group_over_the_deterministic_size_limit():
    intro = _block("text", "Introduction :", order=0)
    group = _block("list", "", order=1, self_ref="#/groups/list")
    items = [
        _block(
            "list_item",
            "x" * 700,
            order=index + 2,
            self_ref=f"#/texts/{index}",
            parent_ref="#/groups/list",
        )
        for index in range(3)
    ]

    assert _composites([intro, group, *items]) == []

    short_items = [
        _block(
            "list_item",
            f"Élément {index}",
            order=index + 2,
            self_ref=f"#/texts/short-{index}",
            parent_ref="#/groups/list",
        )
        for index in range(6)
    ]
    assert _composites([intro, group, *short_items]) == []


def test_v4_is_deterministic_and_does_not_mutate_v1_v2_v3():
    blocks = [
        _block("section_header", "Titre", order=0),
        _block("text", "Intro", order=1),
        _block("text", "Un", order=2),
        _block("text", "Deux", order=3),
    ]
    v1 = build_docling_corpus(blocks)
    v2 = build_structural_docling_corpus(blocks)
    v3 = build_native_docling_corpus(blocks)

    first = build_local_semantic_docling_corpus(blocks)
    second = build_local_semantic_docling_corpus(blocks)

    assert first == second
    assert build_docling_corpus(blocks) == v1
    assert build_structural_docling_corpus(blocks) == v2
    assert build_native_docling_corpus(blocks) == v3
