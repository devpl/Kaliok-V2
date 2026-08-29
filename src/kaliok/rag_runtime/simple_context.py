from __future__ import annotations

from collections.abc import Sequence

from kaliok.rag.types import ContextBundle, RankedCandidate


class RankedContextBuilder:
    def build(
        self,
        question: str,
        candidates: Sequence[RankedCandidate],
    ) -> ContextBundle:
        passages: list[str] = []
        for candidate in candidates:
            provenance = candidate.unit.provenance
            source_unit_id = provenance.metadata.get("source_unit_id")
            passages.append(
                f"[rang={candidate.rank}; source_unit_id={source_unit_id}; "
                f"document_version_id={provenance.document_version_id}]\n"
                f"{candidate.unit.text}"
            )
        return ContextBundle(
            question=question,
            text="\n\n".join(passages),
            candidates=tuple(candidates),
        )
