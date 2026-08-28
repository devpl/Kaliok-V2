from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlmodel import Session

from kaliok.experiments.docling_retrieval import (
    DoclingNativeCorpus,
    DoclingRetrievalUnit,
    DoclingSourceBlock,
    DoclingV4Corpus,
    build_docling_corpus,
    build_local_semantic_docling_corpus,
    build_native_docling_corpus,
    load_docling_source_blocks,
)
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import DocumentVersion


DEFAULT_RUN_IDS = (
    UUID("b9a5bde4-e094-4d52-abe6-3413bdecfe36"),
    UUID("43e7142a-1969-40ba-a9fe-b33650404901"),
)
STRUCTURAL_PARENT_TYPES = {"key_value_area", "list", "picture"}
BOUNDARY_TYPES = {"section_header", "page_header", "page_footer"}
INTRODUCTION_TERMS = re.compile(
    r"\b(objectifs?|missions?|etapes?|criteres?|recommandations?|"
    r"mesures?|conditions?|avantages?|elements?|points?|principes?|"
    r"axes?|priorites?|categories?|constats?|resultats?|actions?)\b"
)
ANNOUNCEMENT_TERMS = re.compile(
    r"\b(sont|suivants?|comprend|comprennent|comporte|comportent|"
    r"vise|visent|distingue|distinguent|triple|principales?)\b"
)


@dataclass(frozen=True)
class StructuralPattern:
    document: str
    page: int
    section: str | None
    introduction: DoclingSourceBlock
    elements: tuple[DoclingSourceBlock, ...]
    relation_parents: tuple[DoclingSourceBlock, ...]
    classifications: tuple[str, ...]
    reconstructible_from_relations: bool


@dataclass(frozen=True)
class UnitCoverage:
    introduction_alone: bool
    element_alone: bool
    complete: bool
    partial: bool
    fragmented: bool


@dataclass(frozen=True)
class AuditedPattern:
    pattern: StructuralPattern
    v1: UnitCoverage
    v3: UnitCoverage
    v4: UnitCoverage


@dataclass(frozen=True)
class DocumentAudit:
    document: str
    processing_run_id: UUID
    patterns: tuple[AuditedPattern, ...]
    v4_corpus: DoclingV4Corpus

    @property
    def complete_v1(self) -> int:
        return sum(pattern.v1.complete for pattern in self.patterns)

    @property
    def complete_v3(self) -> int:
        return sum(pattern.v3.complete for pattern in self.patterns)

    @property
    def fragmented_v1(self) -> int:
        return sum(pattern.v1.fragmented for pattern in self.patterns)

    @property
    def fragmented_v3(self) -> int:
        return sum(pattern.v3.fragmented for pattern in self.patterns)

    @property
    def complete_v4(self) -> int:
        return sum(pattern.v4.complete for pattern in self.patterns)

    @property
    def fragmented_v4(self) -> int:
        return sum(pattern.v4.fragmented for pattern in self.patterns)

    @property
    def reconstructible(self) -> int:
        return sum(
            pattern.pattern.reconstructible_from_relations
            for pattern in self.patterns
        )


def normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def is_structural_introduction(block: DoclingSourceBlock) -> bool:
    if (
        not block.indexable
        or block.content_layer == "furniture"
        or block.block_type in BOUNDARY_TYPES
    ):
        return False
    text = normalized_text(block.content)
    if not text or len(text) > 600:
        return False
    has_structure_term = INTRODUCTION_TERMS.search(text) is not None
    ends_with_colon = block.content.rstrip().endswith(":")
    if ends_with_colon:
        return bool(has_structure_term)
    if len(block.content) > 180:
        return False
    announces = ANNOUNCEMENT_TERMS.search(text) is not None
    return bool(has_structure_term and announces)


def detect_structural_patterns(
    document: str,
    blocks: list[DoclingSourceBlock],
) -> list[StructuralPattern]:
    ordered = sorted(
        blocks,
        key=lambda block: (block.page, block.reading_order, str(block.block_id)),
    )
    by_ref = {
        block.self_ref: block for block in ordered if block.self_ref is not None
    }
    section_by_block: dict[UUID, str | None] = {}
    current_section: str | None = None
    for block in ordered:
        if block.block_type == "section_header" and block.content.strip():
            current_section = block.content.strip()
        section_by_block[block.block_id] = current_section

    patterns: list[StructuralPattern] = []
    consumed_introductions: set[UUID] = set()
    for index, introduction in enumerate(ordered):
        if (
            introduction.block_id in consumed_introductions
            or not is_structural_introduction(introduction)
        ):
            continue
        elements, explicit_parent = _associated_elements(
            introduction,
            ordered,
            index,
            by_ref,
        )
        if len(elements) < 2 and not any(
            element.block_type == "table" for element in elements
        ):
            continue
        classifications = classify_pattern(
            introduction,
            elements,
            by_ref,
        )
        relation_parent_refs = {
            block.parent_ref
            for block in (introduction, *elements)
            if block.parent_ref in by_ref
        }
        relation_parents = tuple(
            sorted(
                (by_ref[reference] for reference in relation_parent_refs),
                key=lambda block: (
                    block.reading_order,
                    str(block.block_id),
                ),
            )
        )
        patterns.append(
            StructuralPattern(
                document=document,
                page=introduction.page,
                section=section_by_block[introduction.block_id],
                introduction=introduction,
                elements=tuple(elements),
                relation_parents=relation_parents,
                classifications=classifications,
                reconstructible_from_relations=explicit_parent,
            )
        )
        consumed_introductions.add(introduction.block_id)
    return patterns


def _associated_elements(
    introduction: DoclingSourceBlock,
    ordered: list[DoclingSourceBlock],
    index: int,
    by_ref: dict[str, DoclingSourceBlock],
) -> tuple[list[DoclingSourceBlock], bool]:
    direct = [
        by_ref[reference]
        for reference in introduction.children_refs
        if reference in by_ref
    ]
    if not direct and introduction.self_ref is not None:
        direct = [
            block
            for block in ordered
            if block.parent_ref == introduction.self_ref
        ]
    direct = _usable_elements(direct, introduction.page)
    if direct:
        return direct, True

    parent = (
        by_ref.get(introduction.parent_ref)
        if introduction.parent_ref is not None
        else None
    )
    if parent is not None and parent.block_type in STRUCTURAL_PARENT_TYPES:
        siblings = [
            by_ref[reference]
            for reference in parent.children_refs
            if reference in by_ref and reference != introduction.self_ref
        ]
        siblings = _usable_elements(siblings, introduction.page)
        if siblings:
            return siblings, True

    adjacent: list[DoclingSourceBlock] = []
    structural_element_parent: str | None = None
    for candidate in ordered[index + 1 :]:
        if candidate.page != introduction.page:
            break
        if candidate.block_type in BOUNDARY_TYPES:
            break
        if is_structural_introduction(candidate):
            break
        if (
            not candidate.indexable
            or candidate.content_layer == "furniture"
            or not candidate.content.strip()
        ):
            continue
        if structural_element_parent is not None:
            if candidate.parent_ref != structural_element_parent:
                break
        elif candidate.parent_ref != introduction.parent_ref:
            if candidate.block_type == "list_item":
                structural_element_parent = candidate.parent_ref
            elif candidate.block_type != "table":
                break
        adjacent.append(candidate)
        if candidate.block_type == "table" or len(adjacent) >= 10:
            break
    return adjacent, False


def _usable_elements(
    blocks: list[DoclingSourceBlock],
    page: int,
) -> list[DoclingSourceBlock]:
    return sorted(
        (
            block
            for block in blocks
            if block.page == page
            and block.indexable
            and block.content_layer != "furniture"
            and block.block_type not in BOUNDARY_TYPES
            and block.content.strip()
        ),
        key=lambda block: (block.reading_order, str(block.block_id)),
    )


def classify_pattern(
    introduction: DoclingSourceBlock,
    elements: list[DoclingSourceBlock],
    by_ref: dict[str, DoclingSourceBlock],
) -> tuple[str, ...]:
    categories: set[str] = set()
    element_types = {element.block_type for element in elements}
    parent = by_ref.get(introduction.parent_ref or "")
    if "list_item" in element_types or (
        parent is not None and parent.block_type == "list"
    ):
        categories.add("LISTE")
    if parent is not None and parent.block_type in {
        "key_value_area",
        "list",
    }:
        categories.add("GROUPE")
    if "table" in element_types:
        categories.add("TABLE")
    all_blocks = [introduction, *elements]
    if any(
        (block.parent_ref or "").startswith("#/pictures/")
        for block in all_blocks
    ):
        categories.add("PICTURE_TEXT")
    if introduction.block_type == "section_header":
        categories.add("TITRE + CONTENU")
    if (
        len(elements) >= 2
        and all(
            element.parent_ref == introduction.parent_ref
            for element in elements
        )
        and (parent is None or parent.block_type not in STRUCTURAL_PARENT_TYPES)
    ):
        categories.add("PARAGRAPHES FRÈRES")
    if not categories:
        categories.add("AUTRE")
    return tuple(sorted(categories))


def audit_patterns(
    patterns: list[StructuralPattern],
    v1_units: list[DoclingRetrievalUnit],
    v3_corpus: DoclingNativeCorpus,
    v4_corpus: DoclingV4Corpus,
) -> list[AuditedPattern]:
    return [
        AuditedPattern(
            pattern=pattern,
            v1=_coverage_for_units(pattern, v1_units),
            v3=_coverage_for_units(pattern, v3_corpus.units),
            v4=_coverage_for_units(pattern, v4_corpus.units),
        )
        for pattern in patterns
    ]


def _coverage_for_units(pattern: StructuralPattern, units) -> UnitCoverage:
    introduction_text = normalized_text(pattern.introduction.content)
    element_texts = [
        normalized_text(element.content) for element in pattern.elements
    ]
    required_ids = {
        pattern.introduction.block_id,
        *(element.block_id for element in pattern.elements),
    }
    introduction_covered = False
    covered_elements: set[int] = set()
    introduction_alone = False
    element_alone = False
    partial = False
    complete = False

    for unit in units:
        unit_text = normalized_text(unit.text)
        source_ids = set(
            getattr(
                unit,
                "source_block_ids",
                (getattr(unit, "source_block_id", None),),
            )
        )
        source_ids.discard(None)
        has_introduction = (
            pattern.introduction.block_id in source_ids
            or bool(introduction_text and introduction_text in unit_text)
        )
        matched_elements = {
            index
            for index, (element, text) in enumerate(
                zip(pattern.elements, element_texts, strict=True)
            )
            if element.block_id in source_ids or bool(text and text in unit_text)
        }
        introduction_covered |= has_introduction
        covered_elements.update(matched_elements)
        complete |= required_ids.issubset(source_ids) or (
            has_introduction and len(matched_elements) == len(element_texts)
        )
        introduction_alone |= has_introduction and not matched_elements
        element_alone |= not has_introduction and len(matched_elements) == 1
        partial |= bool(matched_elements) and (
            has_introduction or len(matched_elements) > 1
        ) and not (
            has_introduction and len(matched_elements) == len(element_texts)
        )

    fragmented = bool(
        not complete
        and introduction_covered
        and len(covered_elements) == len(element_texts)
    )
    return UnitCoverage(
        introduction_alone=introduction_alone,
        element_alone=element_alone,
        complete=complete,
        partial=partial,
        fragmented=fragmented,
    )


def audit_run(session: Session, run_id: UUID) -> DocumentAudit:
    run, blocks = load_docling_source_blocks(session, run_id)
    version = session.get(DocumentVersion, run.document_version_id)
    if version is None:
        raise ValueError(f"DocumentVersion introuvable pour le run {run_id}.")
    patterns = detect_structural_patterns(version.filename, blocks)
    v1 = build_docling_corpus(blocks)
    v3 = build_native_docling_corpus(blocks)
    v4 = build_local_semantic_docling_corpus(blocks)
    return DocumentAudit(
        document=version.filename,
        processing_run_id=run_id,
        patterns=tuple(audit_patterns(patterns, v1.units, v3, v4)),
        v4_corpus=v4,
    )


def print_report(audits: list[DocumentAudit]) -> None:
    for audit in audits:
        print()
        print("=" * 88)
        print(f"DOCUMENT : {audit.document}")
        print(f"RUN      : {audit.processing_run_id}")
        print("=" * 88)
        for number, audited in enumerate(audit.patterns, start=1):
            pattern = audited.pattern
            intro = pattern.introduction
            print()
            print(f"CAS {number} | page {pattern.page} | {', '.join(pattern.classifications)}")
            print(f"Section            : {pattern.section!r}")
            print(f"Type introduction  : {intro.block_type}")
            print(f"Introduction       : {intro.content}")
            print(
                "Références intro    : "
                f"self={intro.self_ref!r} parent={intro.parent_ref!r} "
                f"children={list(intro.children_refs)}"
            )
            print(f"Éléments associés  : {len(pattern.elements)}")
            print(
                "Types éléments      : "
                f"{[element.block_type for element in pattern.elements]}"
            )
            for element_number, element in enumerate(pattern.elements, start=1):
                print(
                    f"  [{element_number}] {element.content}\n"
                    f"      self={element.self_ref!r} "
                    f"parent={element.parent_ref!r} "
                    f"children={list(element.children_refs)}"
                )
            if pattern.relation_parents:
                print("Parents structurels :")
                for parent in pattern.relation_parents:
                    print(
                        f"  type={parent.block_type} self={parent.self_ref!r} "
                        f"parent={parent.parent_ref!r} "
                        f"children={list(parent.children_refs)}"
                    )
            print(
                "Relations suffisantes: "
                f"{'oui' if pattern.reconstructible_from_relations else 'non'}"
            )
            _print_coverage("V1", audited.v1)
            _print_coverage("V3", audited.v3)
            _print_coverage("V4", audited.v4)
        _print_document_statistics(audit)

    print()
    print("=" * 88)
    print("RÉCAPITULATIF MULTI-DOCUMENTS")
    print("=" * 88)
    print(f"Documents audités                  : {len(audits)}")
    print(f"Motifs détectés                    : {sum(len(a.patterns) for a in audits)}")
    print(f"Entièrement reconstruits par V1    : {sum(a.complete_v1 for a in audits)}")
    print(f"Entièrement reconstruits par V3    : {sum(a.complete_v3 for a in audits)}")
    print(f"Entièrement reconstruits par V4    : {sum(a.complete_v4 for a in audits)}")
    print(f"Fragmentés dans V1                 : {sum(a.fragmented_v1 for a in audits)}")
    print(f"Fragmentés dans V3                 : {sum(a.fragmented_v3 for a in audits)}")
    print(f"Fragmentés dans V4                 : {sum(a.fragmented_v4 for a in audits)}")
    print(f"Reconstructibles via les relations : {sum(a.reconstructible for a in audits)}")


def _print_coverage(label: str, coverage: UnitCoverage) -> None:
    print(
        f"{label:<20}: introduction_seule={coverage.introduction_alone} "
        f"element_seul={coverage.element_alone} "
        f"introduction_tous_elements={coverage.complete} "
        f"groupe_partiel={coverage.partial} fragmente={coverage.fragmented}"
    )


def _print_document_statistics(audit: DocumentAudit) -> None:
    print()
    print("STATISTIQUES")
    print(f"Motifs détectés                    : {len(audit.patterns)}")
    print(f"Entièrement reconstruits par V1    : {audit.complete_v1}")
    print(f"Entièrement reconstruits par V3    : {audit.complete_v3}")
    print(f"Entièrement reconstruits par V4    : {audit.complete_v4}")
    print(f"Fragmentés dans V1                 : {audit.fragmented_v1}")
    print(f"Fragmentés dans V3                 : {audit.fragmented_v3}")
    print(f"Fragmentés dans V4                 : {audit.fragmented_v4}")
    print(f"Reconstructibles via les relations : {audit.reconstructible}")
    lengths = audit.v4_corpus.lengths
    source_counts = audit.v4_corpus.source_blocks_per_composite
    print(f"Unités V4                          : {len(audit.v4_corpus.units)}")
    print(f"Unités composites V4               : {audit.v4_corpus.composite_count}")
    print(f"Unités atomiques V4                : {audit.v4_corpus.atomic_count}")
    print(f"Composites par page                : {audit.v4_corpus.composites_by_page}")
    print(
        "Longueur V4 moyenne/min/max        : "
        f"{(sum(lengths) / len(lengths)) if lengths else 0:.1f}/"
        f"{min(lengths) if lengths else 0}/{max(lengths) if lengths else 0}"
    )
    print(
        "Blocs sources/composite moy/min/max: "
        f"{(sum(source_counts) / len(source_counts)) if source_counts else 0:.1f}/"
        f"{min(source_counts) if source_counts else 0}/"
        f"{max(source_counts) if source_counts else 0}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit structurel en lecture seule des ContentBlock Docling."
        )
    )
    parser.add_argument(
        "--docling-run-id",
        action="append",
        type=UUID,
        dest="run_ids",
        help=(
            "Run Docling completed à auditer. Répétable. Par défaut : "
            "runs RIDEAU et CTC connus."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_ids = tuple(arguments.run_ids or DEFAULT_RUN_IDS)
    engine = create_database_engine()
    with Session(engine) as session:
        audits = [audit_run(session, run_id) for run_id in run_ids]
    audits.sort(key=lambda audit: (audit.document, str(audit.processing_run_id)))
    print_report(audits)


if __name__ == "__main__":
    main()
