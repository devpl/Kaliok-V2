from uuid import UUID

from kaliok.retrieval.hybrid import (
    search_hybrid_chunks,
)


DOCUMENT_VERSION_ID = UUID(
    "6d7dfaeb-e504-467d-917e-0c7e745bd18e"
)

EMBEDDING_MODEL_ID = UUID(
    "46164d84-fa91-423e-a375-27472892acb9"
)


QUESTIONS = [
    "Quelle est la durée de fabrication ?",
    "Quel est le prix TTC ?",
    "Quelles sont les conditions de paiement ?",
    "Quelle est la dimension de la véranda ?",
    "Quelle est la garantie indiquée ?",
]


def main() -> None:
    for question_number, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print()
        print("=" * 70)
        print(
            f"QUESTION {question_number}"
        )
        print("=" * 70)
        print(question)

        results = search_hybrid_chunks(
            question,
            embedding_model_id=EMBEDDING_MODEL_ID,
            document_version_id=DOCUMENT_VERSION_ID,
            limit=5,
            candidate_limit=10,
            rrf_k=60,
        )

        if not results:
            print()
            print(
                "Aucun résultat hybride."
            )
            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(
                f"#{rank} "
                f"| rrf={result.rrf_score:.6f} "
                f"| vector_rank={result.vector_rank} "
                f"| lexical_rank={result.lexical_rank}"
            )

            if result.vector_distance is not None:
                print(
                    "   vector_distance="
                    f"{result.vector_distance:.4f}"
                )

            if result.lexical_score is not None:
                print(
                    "   lexical_score="
                    f"{result.lexical_score:.4f}"
                )

            print(
                f"   chunk={result.chunk_id}"
            )

            print(
                result.content
            )


if __name__ == "__main__":
    main()
