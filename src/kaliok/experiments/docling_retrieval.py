from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from kaliok.documents.chunking import split_long_text
from kaliok.storage.models import ContentBlock, Page, ProcessingRun


DEFAULT_MAX_CHARS = 1000
STRUCTURAL_TARGET_CHARS = 400
STRUCTURAL_MAX_CHARS = 700
V4_MAX_COMPOSITE_CHARS = 2000
V4_MAX_INTRO_CHARS = 240
V4_MAX_GROUP_BLOCKS = 6
V4_MIN_SIBLING_BLOCKS = 3
V4_MAX_SIBLING_BLOCKS = 5
EXCLUDED_BLOCK_TYPES = {"page_header", "page_footer"}
MERGEABLE_BLOCK_TYPES = {"text", "list_item", "key_value_area"}


@dataclass(frozen=True)
class DoclingSourceBlock:
    block_id: UUID
    page: int
    block_type: str
    content: str
    reading_order: int
    indexable: bool
    content_layer: str | None
    self_ref: str | None = None
    parent_ref: str | None = None
    children_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DoclingRetrievalUnit:
    source_block_id: UUID
    page: int
    block_type: str
    section_header: str | None
    text: str


@dataclass(frozen=True)
class DoclingCorpus:
    source_block_count: int
    excluded_block_count: int
    units: list[DoclingRetrievalUnit]

    @property
    def units_by_type(self) -> dict[str, int]:
        return dict(Counter(unit.block_type for unit in self.units))

    @property
    def units_by_page(self) -> dict[int, int]:
        return dict(Counter(unit.page for unit in self.units))

    @property
    def lengths(self) -> list[int]:
        return [len(unit.text) for unit in self.units]


@dataclass(frozen=True)
class DoclingSearchResult:
    unit: (
        DoclingRetrievalUnit
        | DoclingStructuralUnit
        | DoclingNativeUnit
        | DoclingV4Unit
    )
    distance: float


@dataclass(frozen=True)
class RankedPage:
    page: int
    rank: int


@dataclass(frozen=True)
class FusedPage:
    page: int
    score: float


@dataclass(frozen=True)
class DoclingStructuralUnit:
    page: int
    section_header: str | None
    source_block_types: tuple[str, ...]
    source_block_ids: tuple[UUID, ...]
    text: str


@dataclass(frozen=True)
class DoclingStructuralCorpus:
    source_block_count: int
    excluded_block_count: int
    units: list[DoclingStructuralUnit]

    @property
    def units_by_page(self) -> dict[int, int]:
        return dict(Counter(unit.page for unit in self.units))

    @property
    def lengths(self) -> list[int]:
        return [len(unit.text) for unit in self.units]


@dataclass(frozen=True)
class DoclingNativeUnit:
    page: int
    logical_type: str
    section_header: str | None
    parent_ref: str | None
    source_block_types: tuple[str, ...]
    source_block_ids: tuple[UUID, ...]
    text: str


@dataclass(frozen=True)
class DoclingNativeCorpus:
    source_block_count: int
    excluded_block_count: int
    units: list[DoclingNativeUnit]

    @property
    def units_by_type(self) -> dict[str, int]:
        return dict(Counter(unit.logical_type for unit in self.units))

    @property
    def units_by_page(self) -> dict[int, int]:
        return dict(Counter(unit.page for unit in self.units))

    @property
    def lengths(self) -> list[int]:
        return [len(unit.text) for unit in self.units]


@dataclass(frozen=True)
class DoclingV4Unit:
    page: int
    logical_type: str
    section_header: str | None
    parent_ref: str | None
    group_ref: str | None
    parent_refs: tuple[str, ...]
    source_block_types: tuple[str, ...]
    source_block_ids: tuple[UUID, ...]
    text: str
    is_composite: bool


@dataclass(frozen=True)
class DoclingV4Corpus:
    source_block_count: int
    excluded_block_count: int
    units: list[DoclingV4Unit]

    @property
    def units_by_type(self) -> dict[str, int]:
        return dict(Counter(unit.logical_type for unit in self.units))

    @property
    def units_by_page(self) -> dict[int, int]:
        return dict(Counter(unit.page for unit in self.units))

    @property
    def lengths(self) -> list[int]:
        return [len(unit.text) for unit in self.units]

    @property
    def composite_count(self) -> int:
        return sum(unit.is_composite for unit in self.units)

    @property
    def atomic_count(self) -> int:
        return len(self.units) - self.composite_count

    @property
    def composites_by_page(self) -> dict[int, int]:
        return dict(
            Counter(unit.page for unit in self.units if unit.is_composite)
        )

    @property
    def source_blocks_per_composite(self) -> list[int]:
        return [
            len(unit.source_block_ids)
            for unit in self.units
            if unit.is_composite
        ]


def load_docling_source_blocks(
    session: Session,
    processing_run_id: UUID,
) -> tuple[ProcessingRun, list[DoclingSourceBlock]]:
    run = session.exec(
        select(ProcessingRun).where(
            ProcessingRun.id == processing_run_id,
            ProcessingRun.engine == "docling",
            ProcessingRun.status == "completed",
        )
    ).one_or_none()
    if run is None:
        raise ValueError(
            "ProcessingRun Docling completed introuvable : "
            f"{processing_run_id}."
        )

    rows = session.exec(
        select(ContentBlock, Page)
        .join(Page, Page.id == ContentBlock.page_id)
        .where(ContentBlock.processing_run_id == run.id)
    ).all()
    blocks = [
        DoclingSourceBlock(
            block_id=block.id,
            page=page.page_number,
            block_type=block.block_type,
            content=block.content,
            reading_order=(
                block.reading_order
                if block.reading_order is not None
                else block.block_index
            ),
            indexable=bool(block.extra_data.get("indexable", True)),
            content_layer=block.extra_data.get("content_layer"),
            self_ref=block.extra_data.get("docling_self_ref"),
            parent_ref=block.extra_data.get("docling_parent_ref"),
            children_refs=tuple(
                block.extra_data.get("docling_children", [])
            ),
        )
        for block, page in rows
    ]
    blocks.sort(
        key=lambda block: (block.reading_order, str(block.block_id))
    )
    return run, blocks


def build_docling_corpus(
    blocks: list[DoclingSourceBlock],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> DoclingCorpus:
    if max_chars <= 0:
        raise ValueError("max_chars doit être strictement positif.")

    units: list[DoclingRetrievalUnit] = []
    excluded = 0
    current_section: str | None = None

    for block in blocks:
        content = block.content.strip()
        if (
            not block.indexable
            or block.block_type in EXCLUDED_BLOCK_TYPES
            or block.content_layer == "furniture"
            or not content
        ):
            excluded += 1
            continue

        if block.block_type == "section_header":
            current_section = content
            parts = split_long_text(content, max_chars)
            units.extend(
                DoclingRetrievalUnit(
                    source_block_id=block.block_id,
                    page=block.page,
                    block_type=block.block_type,
                    section_header=current_section,
                    text=part,
                )
                for part in parts
            )
            continue

        prefix = f"{current_section}\n" if current_section else ""
        available = max_chars - len(prefix)
        if available <= 0:
            header_budget = max_chars // 2
            prefix = (
                f"{current_section[:header_budget]}\n"
                if header_budget
                else ""
            )
            available = max_chars - len(prefix)

        parts = split_long_text(content, available)
        units.extend(
            DoclingRetrievalUnit(
                source_block_id=block.block_id,
                page=block.page,
                block_type=block.block_type,
                section_header=current_section,
                text=f"{prefix}{part}",
            )
            for part in parts
        )

    return DoclingCorpus(
        source_block_count=len(blocks),
        excluded_block_count=excluded,
        units=units,
    )


def build_structural_docling_corpus(
    blocks: list[DoclingSourceBlock],
    *,
    target_chars: int = STRUCTURAL_TARGET_CHARS,
    max_chars: int = STRUCTURAL_MAX_CHARS,
) -> DoclingStructuralCorpus:
    if target_chars <= 0 or max_chars <= 0 or target_chars > max_chars:
        raise ValueError("Limites structurelles Docling invalides.")

    units: list[DoclingStructuralUnit] = []
    excluded = 0
    current_section: str | None = None
    pending: list[DoclingSourceBlock] = []

    def unit_text(parts: list[DoclingSourceBlock]) -> str:
        body = "\n".join(part.content.strip() for part in parts)
        return f"{current_section}\n{body}" if current_section else body

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        units.append(
            DoclingStructuralUnit(
                page=pending[0].page,
                section_header=current_section,
                source_block_types=tuple(
                    block.block_type for block in pending
                ),
                source_block_ids=tuple(block.block_id for block in pending),
                text=unit_text(pending),
            )
        )
        pending = []

    def add_isolated(block: DoclingSourceBlock) -> None:
        prefix = f"{current_section}\n" if current_section else ""
        available = max_chars - len(prefix)
        if available <= 0:
            header_budget = max_chars // 2
            prefix = (
                f"{current_section[:header_budget]}\n"
                if header_budget
                else ""
            )
            available = max_chars - len(prefix)
        for part in split_long_text(block.content.strip(), available):
            units.append(
                DoclingStructuralUnit(
                    page=block.page,
                    section_header=current_section,
                    source_block_types=(block.block_type,),
                    source_block_ids=(block.block_id,),
                    text=f"{prefix}{part}",
                )
            )

    for block in blocks:
        content = block.content.strip()
        if (
            not block.indexable
            or block.block_type in EXCLUDED_BLOCK_TYPES
            or block.content_layer == "furniture"
            or not content
        ):
            flush()
            excluded += 1
            continue

        if block.block_type == "section_header":
            flush()
            current_section = content
            continue

        if block.block_type not in MERGEABLE_BLOCK_TYPES:
            flush()
            add_isolated(block)
            continue

        if pending and (
            pending[0].page != block.page
            or len(unit_text([*pending, block])) > max_chars
        ):
            flush()

        if len(unit_text([block])) > max_chars:
            add_isolated(block)
            continue

        pending.append(block)
        if len(unit_text(pending)) >= target_chars:
            flush()

    flush()
    return DoclingStructuralCorpus(
        source_block_count=len(blocks),
        excluded_block_count=excluded,
        units=units,
    )


def build_native_docling_corpus(
    blocks: list[DoclingSourceBlock],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> DoclingNativeCorpus:
    if max_chars <= 0:
        raise ValueError("max_chars doit être strictement positif.")

    by_ref = {
        block.self_ref: block for block in blocks if block.self_ref is not None
    }
    children_by_parent: dict[str, list[DoclingSourceBlock]] = {}
    for block in blocks:
        if block.parent_ref is not None:
            children_by_parent.setdefault(block.parent_ref, []).append(block)

    for children in children_by_parent.values():
        children.sort(
            key=lambda child: (child.reading_order, str(child.block_id))
        )

    units: list[DoclingNativeUnit] = []
    consumed_refs: set[str] = set()
    excluded_refs: set[str] = set()
    current_section: str | None = None

    def usable(block: DoclingSourceBlock) -> bool:
        return bool(
            block.indexable
            and block.block_type not in EXCLUDED_BLOCK_TYPES
            and block.content_layer != "furniture"
            and block.content.strip()
        )

    def ordered_children(parent: DoclingSourceBlock) -> list[DoclingSourceBlock]:
        referenced = [
            by_ref[ref]
            for ref in parent.children_refs
            if ref in by_ref
        ]
        candidates = (
            referenced
            if referenced
            else children_by_parent.get(parent.self_ref or "", [])
        )
        return [child for child in candidates if usable(child)]

    structural_types = {"key_value_area", "list", "picture"}
    aggregated_children: dict[UUID, list[DoclingSourceBlock]] = {}
    for parent in blocks:
        if parent.block_type not in structural_types:
            continue
        children = ordered_children(parent)
        if not children:
            continue
        aggregated_children[parent.block_id] = children
        consumed_refs.update(
            child.self_ref
            for child in children
            if child.self_ref is not None
        )

    def append_unit(
        *,
        logical_type: str,
        page: int,
        parent_ref: str | None,
        sources: list[DoclingSourceBlock],
        body: str,
    ) -> None:
        prefix = f"{current_section}\n" if current_section else ""
        available = max_chars - len(prefix)
        if available <= 0:
            header_budget = max_chars // 2
            prefix = (
                f"{current_section[:header_budget]}\n"
                if header_budget
                else ""
            )
            available = max_chars - len(prefix)
        for part in split_long_text(body, available):
            units.append(
                DoclingNativeUnit(
                    page=page,
                    logical_type=logical_type,
                    section_header=current_section,
                    parent_ref=parent_ref,
                    source_block_types=tuple(
                        source.block_type for source in sources
                    ),
                    source_block_ids=tuple(
                        source.block_id for source in sources
                    ),
                    text=f"{prefix}{part}",
                )
            )

    for block in blocks:
        if block.self_ref in consumed_refs:
            continue
        if not usable(block) and block.block_type not in structural_types:
            if block.self_ref is not None:
                excluded_refs.add(block.self_ref)
            continue

        if block.block_type == "section_header":
            current_section = block.content.strip()
            units.append(
                DoclingNativeUnit(
                    page=block.page,
                    logical_type="section_header",
                    section_header=current_section,
                    parent_ref=block.parent_ref,
                    source_block_types=(block.block_type,),
                    source_block_ids=(block.block_id,),
                    text=current_section,
                )
            )
            continue

        if block.block_type in structural_types:
            children = aggregated_children.get(block.block_id, [])
            if not children:
                if block.self_ref is not None:
                    excluded_refs.add(block.self_ref)
                continue
            separator = " " if block.block_type == "key_value_area" else "\n"
            logical_type = (
                "picture_text"
                if block.block_type == "picture"
                else block.block_type
            )
            append_unit(
                logical_type=logical_type,
                page=block.page,
                parent_ref=block.self_ref,
                sources=children,
                body=separator.join(
                    child.content.strip() for child in children
                ),
            )
            continue

        append_unit(
            logical_type=block.block_type,
            page=block.page,
            parent_ref=block.parent_ref,
            sources=[block],
            body=block.content.strip(),
        )

    excluded_count = sum(
        1
        for block in blocks
        if block.self_ref in excluded_refs
        or block.self_ref in consumed_refs
    )
    return DoclingNativeCorpus(
        source_block_count=len(blocks),
        excluded_block_count=excluded_count,
        units=units,
    )


def build_local_semantic_docling_corpus(
    blocks: list[DoclingSourceBlock],
    *,
    max_composite_chars: int = V4_MAX_COMPOSITE_CHARS,
    max_intro_chars: int = V4_MAX_INTRO_CHARS,
    max_group_blocks: int = V4_MAX_GROUP_BLOCKS,
    min_sibling_blocks: int = V4_MIN_SIBLING_BLOCKS,
    max_sibling_blocks: int = V4_MAX_SIBLING_BLOCKS,
) -> DoclingV4Corpus:
    """Build V3-like atomic units plus bounded local semantic composites."""
    if (
        max_composite_chars <= 0
        or max_intro_chars <= 0
        or max_group_blocks < 2
    ):
        raise ValueError("Limites V4 Docling invalides.")
    if min_sibling_blocks < 2 or max_sibling_blocks < min_sibling_blocks:
        raise ValueError("Bornes de groupe frère V4 invalides.")

    v3 = build_native_docling_corpus(blocks)
    units = [
        DoclingV4Unit(
            page=unit.page,
            logical_type=unit.logical_type,
            section_header=unit.section_header,
            parent_ref=unit.parent_ref,
            group_ref=None,
            parent_refs=(unit.parent_ref,) if unit.parent_ref else (),
            source_block_types=unit.source_block_types,
            source_block_ids=unit.source_block_ids,
            text=unit.text,
            is_composite=False,
        )
        for unit in v3.units
    ]

    ordered = sorted(
        blocks,
        key=lambda block: (
            block.page,
            block.reading_order,
            str(block.block_id),
        ),
    )
    by_ref = {
        block.self_ref: block for block in ordered if block.self_ref is not None
    }
    children_by_parent: dict[str, list[DoclingSourceBlock]] = {}
    for block in ordered:
        if block.parent_ref:
            children_by_parent.setdefault(block.parent_ref, []).append(block)

    def usable(block: DoclingSourceBlock) -> bool:
        return bool(
            block.indexable
            and block.block_type not in EXCLUDED_BLOCK_TYPES
            and block.content_layer != "furniture"
            and block.content.strip()
        )

    sections: dict[UUID, str | None] = {}
    current_page: int | None = None
    current_section: str | None = None
    for block in ordered:
        if block.page != current_page:
            current_page = block.page
            current_section = None
        if block.block_type == "section_header" and usable(block):
            current_section = block.content.strip()
        sections[block.block_id] = current_section

    content_order = [
        block
        for block in ordered
        if usable(block) and block.block_type != "section_header"
    ]
    content_position = {
        block.block_id: position
        for position, block in enumerate(content_order)
    }
    composite_sources: set[tuple[UUID, ...]] = set()

    def composite_text(
        sources: list[DoclingSourceBlock], section: str | None
    ) -> str:
        body = "\n".join(source.content.strip() for source in sources)
        return f"{section}\n{body}" if section else body

    def add_composite(
        *,
        logical_type: str,
        sources: list[DoclingSourceBlock],
        section: str | None,
        parent_ref: str | None,
        group_ref: str | None,
    ) -> bool:
        source_ids = tuple(source.block_id for source in sources)
        text = composite_text(sources, section)
        if source_ids in composite_sources or len(text) > max_composite_chars:
            return False
        composite_sources.add(source_ids)
        units.append(
            DoclingV4Unit(
                page=sources[0].page,
                logical_type=logical_type,
                section_header=section,
                parent_ref=parent_ref,
                group_ref=group_ref,
                parent_refs=tuple(
                    dict.fromkeys(
                        source.parent_ref
                        for source in sources
                        if source.parent_ref
                    )
                ),
                source_block_types=tuple(
                    source.block_type for source in sources
                ),
                source_block_ids=source_ids,
                text=text,
                is_composite=True,
            )
        )
        return True

    # Case A: the immediately preceding textual block plus a native group.
    for group in ordered:
        if group.block_type not in {"list", "key_value_area"}:
            continue
        referenced = [
            by_ref[ref] for ref in group.children_refs if ref in by_ref
        ]
        children = referenced or children_by_parent.get(group.self_ref or "", [])
        children = sorted(
            (child for child in children if usable(child)),
            key=lambda child: (
                child.page,
                child.reading_order,
                str(child.block_id),
            ),
        )
        if not children or any(
            child.page != children[0].page for child in children
        ):
            continue
        if len(children) + 1 > max_group_blocks:
            continue
        first_position = content_position.get(children[0].block_id)
        if first_position is None or first_position == 0:
            continue
        intro = content_order[first_position - 1]
        section = sections[children[0].block_id]
        if (
            intro.block_type != "text"
            or intro.page != children[0].page
            or sections[intro.block_id] != section
            or intro.block_id in {child.block_id for child in children}
            or len(intro.content.strip()) > max_intro_chars
        ):
            continue
        add_composite(
            logical_type=f"semantic_group_{group.block_type}",
            sources=[intro, *children],
            section=section,
            parent_ref=group.parent_ref,
            group_ref=group.self_ref,
        )

    # Case B: one bounded suffix of each strictly contiguous text-sibling run.
    sibling_run: list[DoclingSourceBlock] = []

    def flush_sibling_run() -> None:
        nonlocal sibling_run
        if len(sibling_run) < min_sibling_blocks:
            sibling_run = []
            return
        first_start = max(0, len(sibling_run) - max_sibling_blocks)
        last_start = len(sibling_run) - min_sibling_blocks
        for start in range(first_start, last_start + 1):
            candidate = sibling_run[start:]
            if (
                len(candidate) <= max_sibling_blocks
                and len(candidate[0].content.strip()) <= max_intro_chars
                and len(composite_text(
                    candidate, sections[candidate[0].block_id]
                )) <= max_composite_chars
            ):
                add_composite(
                    logical_type="semantic_group_siblings",
                    sources=candidate,
                    section=sections[candidate[0].block_id],
                    parent_ref=candidate[0].parent_ref,
                    group_ref=None,
                )
                break
        sibling_run = []

    for block in ordered:
        is_plain_text = (
            usable(block)
            and block.block_type == "text"
            and not (block.parent_ref or "").startswith("#/pictures/")
        )
        if not is_plain_text:
            flush_sibling_run()
            continue
        if sibling_run and (
            block.page != sibling_run[0].page
            or sections[block.block_id] != sections[sibling_run[0].block_id]
            or block.parent_ref != sibling_run[0].parent_ref
        ):
            flush_sibling_run()
        sibling_run.append(block)
    flush_sibling_run()

    return DoclingV4Corpus(
        source_block_count=len(blocks),
        excluded_block_count=v3.excluded_block_count,
        units=units,
    )


def search_docling_corpus(
    corpus: (
        DoclingCorpus
        | DoclingStructuralCorpus
        | DoclingNativeCorpus
        | DoclingV4Corpus
    ),
    embeddings: list[list[float]],
    query_embedding: list[float],
    *,
    limit: int = 10,
) -> list[DoclingSearchResult]:
    if len(embeddings) != len(corpus.units):
        raise ValueError(
            "Nombre d'embeddings différent du nombre d'unités Docling."
        )

    results = [
        DoclingSearchResult(
            unit=unit,
            distance=_cosine_distance(query_embedding, embedding),
        )
        for unit, embedding in zip(corpus.units, embeddings, strict=True)
    ]
    results.sort(key=lambda result: result.distance)
    return results[:limit]


def best_rank_by_page(
    result_pages: list[list[int]],
    *,
    limit: int = 10,
) -> list[RankedPage]:
    best_ranks: dict[int, int] = {}
    for rank, pages in enumerate(result_pages[:limit], start=1):
        for page in pages:
            best_ranks.setdefault(page, rank)
    return [
        RankedPage(page=page, rank=rank)
        for page, rank in sorted(
            best_ranks.items(),
            key=lambda item: (item[1], item[0]),
        )
    ]


def reciprocal_rank_fusion(
    methods: list[list[RankedPage]],
    *,
    rrf_k: int = 60,
) -> list[FusedPage]:
    if rrf_k < 0:
        raise ValueError("rrf_k doit être positif ou nul.")
    scores: dict[int, float] = {}
    for method in methods:
        seen_pages: set[int] = set()
        for result in method:
            if result.page in seen_pages:
                continue
            seen_pages.add(result.page)
            scores[result.page] = scores.get(result.page, 0.0) + (
                1.0 / (rrf_k + result.rank)
            )
    return [
        FusedPage(page=page, score=score)
        for page, score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def expected_page_rank(
    pages: list[RankedPage] | list[FusedPage],
    expected_pages: list[int],
) -> int | None:
    expected = set(expected_pages)
    return next(
        (
            rank
            for rank, result in enumerate(pages, start=1)
            if result.page in expected
        ),
        None,
    )


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Dimensions d'embedding incohérentes.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 1.0
    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
    return 1.0 - similarity
