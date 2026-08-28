from __future__ import annotations

from uuid import uuid4

from kaliok.experiments.docling_retrieval import (
    DoclingCorpus,
    FusedPage,
    RankedPage,
    DoclingRetrievalUnit,
    DoclingSourceBlock,
    build_docling_corpus,
    build_native_docling_corpus,
    build_structural_docling_corpus,
    best_rank_by_page,
    expected_page_rank,
    load_docling_source_blocks,
    reciprocal_rank_fusion,
    search_docling_corpus,
)
from kaliok.storage.models import ContentBlock, Page, ProcessingRun


def _block(
    block_type: str,
    content: str,
    *,
    page: int,
    order: int,
    indexable: bool = True,
    content_layer: str | None = "body",
    self_ref: str | None = None,
    parent_ref: str | None = None,
    children_refs: tuple[str, ...] = (),
) -> DoclingSourceBlock:
    return DoclingSourceBlock(
        block_id=uuid4(),
        page=page,
        block_type=block_type,
        content=content,
        reading_order=order,
        indexable=indexable,
        content_layer=content_layer,
        self_ref=self_ref,
        parent_ref=parent_ref,
        children_refs=children_refs,
    )


def test_docling_units_filter_and_propagate_structural_context():
    blocks = [
        _block("page_header", "En-tête", page=1, order=0),
        _block("section_header", "Section A", page=1, order=1),
        _block("text", "Premier texte", page=1, order=2),
        _block("list_item", "Élément de liste", page=2, order=3),
        _block("table", "Nom | Valeur", page=2, order=4),
        _block("picture", "", page=2, order=5),
        _block("key_value_area", "   ", page=2, order=6),
        _block("section_header", "Section B", page=3, order=7),
        _block("text", "Second texte", page=3, order=8),
        _block(
            "text",
            "Mobilier",
            page=3,
            order=9,
            content_layer="furniture",
        ),
        _block(
            "text",
            "Explicitement exclu",
            page=3,
            order=10,
            indexable=False,
        ),
    ]

    corpus = build_docling_corpus(blocks)

    assert corpus.source_block_count == 11
    assert corpus.excluded_block_count == 5
    assert [unit.block_type for unit in corpus.units] == [
        "section_header",
        "text",
        "list_item",
        "table",
        "section_header",
        "text",
    ]
    assert [unit.page for unit in corpus.units] == [1, 1, 2, 2, 3, 3]
    assert corpus.units[0].text == "Section A"
    assert corpus.units[1].text == "Section A\nPremier texte"
    assert corpus.units[2].text == "Section A\nÉlément de liste"
    assert corpus.units[3].text == "Section A\nNom | Valeur"
    assert corpus.units[4].text == "Section B"
    assert corpus.units[5].text == "Section B\nSecond texte"
    assert corpus.units[5].section_header == "Section B"


def test_long_docling_block_keeps_section_on_every_fragment():
    blocks = [
        _block("section_header", "Titre", page=4, order=0),
        _block("text", "mot " * 40, page=4, order=1),
    ]

    corpus = build_docling_corpus(blocks, max_chars=50)
    fragments = corpus.units[1:]

    assert len(fragments) > 1
    assert all(unit.page == 4 for unit in fragments)
    assert all(unit.text.startswith("Titre\n") for unit in fragments)
    assert all(unit.text.count("Titre") == 1 for unit in fragments)
    assert all(len(unit.text) <= 50 for unit in fragments)


def test_docling_vector_search_is_deterministic_and_in_memory():
    first = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=1,
        block_type="text",
        section_header=None,
        text="premier",
    )
    second = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=2,
        block_type="table",
        section_header="Prix",
        text="Prix\n23 446 euros",
    )
    corpus = DoclingCorpus(
        source_block_count=2,
        excluded_block_count=0,
        units=[first, second],
    )

    results = search_docling_corpus(
        corpus,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        query_embedding=[0.0, 1.0],
    )

    assert [result.unit.page for result in results] == [2, 1]
    assert results[0].distance == 0.0


def test_structural_v2_merges_adjacent_blocks_and_keeps_sources():
    heading = _block("section_header", "Section", page=1, order=0)
    text = _block("text", "Texte court", page=1, order=1)
    item = _block("list_item", "Élément", page=1, order=2)
    key_value = _block("key_value_area", "Clé : valeur", page=1, order=3)

    corpus = build_structural_docling_corpus(
        [heading, text, item, key_value]
    )

    assert len(corpus.units) == 1
    unit = corpus.units[0]
    assert unit.page == 1
    assert unit.section_header == "Section"
    assert unit.text == "Section\nTexte court\nÉlément\nClé : valeur"
    assert unit.source_block_types == (
        "text",
        "list_item",
        "key_value_area",
    )
    assert unit.source_block_ids == (
        text.block_id,
        item.block_id,
        key_value.block_id,
    )


def test_structural_v2_does_not_merge_pages_or_sections():
    blocks = [
        _block("section_header", "Section A", page=1, order=0),
        _block("text", "Page un", page=1, order=1),
        _block("text", "Page deux", page=2, order=2),
        _block("section_header", "Section B", page=2, order=3),
        _block("text", "Nouvelle section", page=2, order=4),
    ]

    corpus = build_structural_docling_corpus(blocks)

    assert [unit.page for unit in corpus.units] == [1, 2, 2]
    assert [unit.section_header for unit in corpus.units] == [
        "Section A",
        "Section A",
        "Section B",
    ]
    assert corpus.units[0].text == "Section A\nPage un"
    assert corpus.units[1].text == "Section A\nPage deux"
    assert corpus.units[2].text == "Section B\nNouvelle section"


def test_structural_v2_respects_maximum_and_repeats_heading():
    blocks = [
        _block("section_header", "Titre", page=1, order=0),
        _block("text", "a" * 360, page=1, order=1),
        _block("text", "b" * 360, page=1, order=2),
    ]

    corpus = build_structural_docling_corpus(blocks)

    assert len(corpus.units) == 2
    assert all(unit.text.startswith("Titre\n") for unit in corpus.units)
    assert all(len(unit.text) <= 700 for unit in corpus.units)
    assert corpus.units[0].source_block_types == ("text",)
    assert corpus.units[1].source_block_types == ("text",)


def test_structural_v2_keeps_table_isolated():
    text_before = _block("text", "Avant", page=6, order=1)
    table = _block("table", "Nom | Montant", page=6, order=2)
    text_after = _block("text", "Après", page=6, order=3)
    blocks = [
        _block("section_header", "Prix", page=6, order=0),
        text_before,
        table,
        text_after,
    ]

    corpus = build_structural_docling_corpus(blocks)

    assert len(corpus.units) == 3
    assert corpus.units[1].source_block_types == ("table",)
    assert corpus.units[1].source_block_ids == (table.block_id,)
    assert corpus.units[1].text == "Prix\nNom | Montant"


def test_building_structural_v2_does_not_change_v1_units():
    blocks = [
        _block("section_header", "Titre", page=1, order=0),
        _block("text", "Un", page=1, order=1),
        _block("text", "Deux", page=1, order=2),
    ]
    before = build_docling_corpus(blocks)

    build_structural_docling_corpus(blocks)
    after = build_docling_corpus(blocks)

    assert after == before
    assert [unit.text for unit in after.units] == [
        "Titre",
        "Titre\nUn",
        "Titre\nDeux",
    ]


def test_native_v3_aggregates_key_value_area_without_child_duplicates():
    heading = _block(
        "section_header",
        "Projet",
        page=1,
        order=0,
        self_ref="#/texts/header",
    )
    parent = _block(
        "key_value_area",
        "",
        page=1,
        order=1,
        self_ref="#/groups/kv",
        children_refs=("#/texts/key", "#/texts/value"),
    )
    key = _block(
        "text",
        "Projet de :",
        page=1,
        order=2,
        self_ref="#/texts/key",
        parent_ref="#/groups/kv",
    )
    value = _block(
        "text",
        "Mme PELLEGRIN",
        page=1,
        order=3,
        self_ref="#/texts/value",
        parent_ref="#/groups/kv",
    )

    corpus = build_native_docling_corpus([heading, parent, key, value])

    assert len(corpus.units) == 2
    unit = corpus.units[1]
    assert unit.logical_type == "key_value_area"
    assert unit.section_header == "Projet"
    assert unit.parent_ref == "#/groups/kv"
    assert unit.text == "Projet\nProjet de : Mme PELLEGRIN"
    assert unit.source_block_ids == (key.block_id, value.block_id)
    assert sum("Mme PELLEGRIN" in item.text for item in corpus.units) == 1


def test_native_v3_aggregates_list_items_with_section_context():
    heading = _block(
        "section_header",
        "Options",
        page=2,
        order=0,
        self_ref="#/texts/header",
    )
    parent = _block(
        "list",
        "",
        page=2,
        order=1,
        self_ref="#/groups/list",
    )
    first = _block(
        "list_item",
        "Serrure trois points",
        page=2,
        order=2,
        self_ref="#/texts/first",
        parent_ref="#/groups/list",
    )
    second = _block(
        "list_item",
        "Commande à distance",
        page=2,
        order=3,
        self_ref="#/texts/second",
        parent_ref="#/groups/list",
    )

    corpus = build_native_docling_corpus([heading, parent, first, second])
    unit = corpus.units[1]

    assert unit.logical_type == "list"
    assert unit.text == (
        "Options\nSerrure trois points\nCommande à distance"
    )
    assert unit.source_block_types == ("list_item", "list_item")
    assert len(corpus.units) == 2


def test_native_v3_keeps_picture_children_in_a_distinct_unit():
    heading = _block(
        "section_header",
        "Perspectives",
        page=11,
        order=0,
        self_ref="#/texts/header",
    )
    child = _block(
        "text",
        "Perspective gauche",
        page=11,
        order=1,
        self_ref="#/texts/picture",
        parent_ref="#/pictures/1",
    )
    picture = _block(
        "picture",
        "",
        page=11,
        order=2,
        self_ref="#/pictures/1",
    )
    body = _block(
        "text",
        "Texte ordinaire",
        page=11,
        order=3,
        self_ref="#/texts/body",
        parent_ref="#/body",
    )

    corpus = build_native_docling_corpus([heading, child, picture, body])

    assert [unit.logical_type for unit in corpus.units] == [
        "section_header",
        "picture_text",
        "text",
    ]
    picture_unit = corpus.units[1]
    assert picture_unit.parent_ref == "#/pictures/1"
    assert picture_unit.source_block_ids == (child.block_id,)
    assert picture_unit.text == "Perspectives\nPerspective gauche"
    assert corpus.units[2].text == "Perspectives\nTexte ordinaire"


def test_native_v3_keeps_table_and_normal_text_as_separate_v1_like_units():
    heading = _block(
        "section_header",
        "Prix",
        page=6,
        order=0,
        self_ref="#/texts/header",
    )
    table = _block(
        "table",
        "Total | 23 446 euros",
        page=6,
        order=1,
        self_ref="#/tables/0",
    )
    first = _block("text", "Premier", page=6, order=2)
    second = _block("text", "Second", page=6, order=3)

    corpus = build_native_docling_corpus(
        [heading, table, first, second]
    )

    assert [unit.logical_type for unit in corpus.units] == [
        "section_header",
        "table",
        "text",
        "text",
    ]
    assert corpus.units[1].text == "Prix\nTotal | 23 446 euros"
    assert corpus.units[2].text == "Prix\nPremier"
    assert corpus.units[3].text == "Prix\nSecond"


def test_building_native_v3_does_not_change_v1_or_v2_units():
    blocks = [
        _block("section_header", "Titre", page=1, order=0),
        _block("text", "Un", page=1, order=1),
        _block("text", "Deux", page=1, order=2),
    ]
    v1_before = build_docling_corpus(blocks)
    v2_before = build_structural_docling_corpus(blocks)

    build_native_docling_corpus(blocks)

    assert build_docling_corpus(blocks) == v1_before
    assert build_structural_docling_corpus(blocks) == v2_before


def test_page_reduction_keeps_only_best_rank_per_page():
    pages = best_rank_by_page([[5], [2, 5], [5], [3]], limit=10)

    assert pages == [
        RankedPage(page=5, rank=1),
        RankedPage(page=2, rank=2),
        RankedPage(page=3, rank=4),
    ]


def test_rrf_fuses_two_and_three_methods_without_double_counting():
    current = [RankedPage(page=5, rank=1), RankedPage(page=2, rank=2)]
    v1 = [
        RankedPage(page=2, rank=1),
        RankedPage(page=5, rank=3),
        RankedPage(page=5, rank=8),
    ]
    v3 = [RankedPage(page=5, rank=2)]

    two_methods = reciprocal_rank_fusion([current, v1])
    three_methods = reciprocal_rank_fusion([current, v1, v3])

    expected_two_page_5 = 1 / 61 + 1 / 63
    assert next(item for item in two_methods if item.page == 5).score == (
        expected_two_page_5
    )
    expected_three_page_5 = expected_two_page_5 + 1 / 62
    assert three_methods[0] == FusedPage(
        page=5,
        score=expected_three_page_5,
    )


def test_rrf_order_is_deterministic_when_scores_are_equal():
    results = reciprocal_rank_fusion(
        [
            [RankedPage(page=5, rank=1)],
            [RankedPage(page=3, rank=1)],
        ]
    )

    assert [result.page for result in results] == [3, 5]


def test_expected_rank_uses_page_level_order_for_metrics():
    page_ranking = [
        RankedPage(page=4, rank=1),
        RankedPage(page=6, rank=2),
        RankedPage(page=9, rank=4),
    ]

    assert expected_page_rank(page_ranking, [6]) == 2
    assert expected_page_rank(page_ranking, [8, 9]) == 3
    assert expected_page_rank(page_ranking, [11]) is None


class _FakeResult:
    def __init__(self, *, one=None, rows=None):
        self._one = one
        self._rows = rows or []

    def one_or_none(self):
        return self._one

    def all(self):
        return self._rows


class _ReadOnlySession:
    def __init__(self, results):
        self.results = list(results)
        self.exec_count = 0

    def exec(self, statement):
        self.exec_count += 1
        return self.results.pop(0)

    def add(self, value):
        raise AssertionError("Écriture SQL interdite dans l'expérience")

    def delete(self, value):
        raise AssertionError("Suppression SQL interdite dans l'expérience")

    def commit(self):
        raise AssertionError("Commit interdit dans l'expérience")


def test_loading_docling_corpus_is_read_only_for_existing_index_state():
    version_id = uuid4()
    run = ProcessingRun(
        document_version_id=version_id,
        process_type="document_extraction",
        status="completed",
        engine="docling",
    )
    page_pointer = uuid4()
    page = Page(
        document_version_id=version_id,
        page_number=6,
        perception_processing_run_id=page_pointer,
    )
    block = ContentBlock(
        page_id=page.id,
        processing_run_id=run.id,
        block_index=0,
        reading_order=0,
        block_type="table",
        content="Prix | 23 446 €",
        extraction_method="structured",
        extra_data={"indexable": True, "content_layer": "body"},
    )
    existing_chunks = [uuid4()]
    existing_embeddings = [uuid4()]
    chunks_before = tuple(existing_chunks)
    embeddings_before = tuple(existing_embeddings)
    session = _ReadOnlySession(
        [
            _FakeResult(one=run),
            _FakeResult(rows=[(block, page)]),
        ]
    )

    loaded_run, source_blocks = load_docling_source_blocks(session, run.id)

    assert loaded_run is run
    assert session.exec_count == 2
    assert source_blocks[0].page == 6
    assert source_blocks[0].block_type == "table"
    assert page.perception_processing_run_id == page_pointer
    assert tuple(existing_chunks) == chunks_before
    assert tuple(existing_embeddings) == embeddings_before
