from uuid import UUID

from kaliok.retrieval.lexical import (
    search_lexical_chunks,
)


DOCUMENT_VERSION_ID = UUID(
    "6d7dfaeb-e504-467d-917e-0c7e745bd18e"
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
        print("=" * 60)
        print(
            f"QUESTION {question_number}"
        )
        print("=" * 60)
        print(question)

        results = search_lexical_chunks(
            question,
            document_version_id=DOCUMENT_VERSION_ID,
            limit=5,
        )

        if not results:
            print()
            print(
                "Aucun résultat lexical."
            )
            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(
                f"#{rank} "
                f"| lexical_rank={result.rank:.4f} "
                f"| chunk={result.chunk_id}"
            )
            print(
                result.content
            )


if __name__ == "__main__":
    main()
