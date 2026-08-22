from uuid import UUID

from sqlmodel import Session, delete

from kaliok.qa.retrieval import (
    answer_question_with_vector_retrieval,
)
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    Question,
    QuestionAttempt,
    QuestionEvidence,
    QuestionFeedback,
)


DOCUMENT_VERSION_ID = UUID(
    "6d7dfaeb-e504-467d-917e-0c7e745bd18e"
)

EMBEDDING_MODEL_ID = UUID(
    "46164d84-fa91-423e-a375-27472892acb9"
)

QUESTION = "Quel est le prix TTC ?"


def main() -> None:
    print(f"Question : {QUESTION}")
    print()

    result = answer_question_with_vector_retrieval(
        QUESTION,
        embedding_model_id=EMBEDDING_MODEL_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        limit=3,
        origin="system",
    )

    print(
        f"Status     : {result['status']}"
    )
    print(
        f"QuestionID : {result['question_id']}"
    )
    print(
        f"AttemptID  : {result['attempt_id']}"
    )

    if "confidence" in result:
        print(
            f"Confidence : {result['confidence']:.4f}"
        )

    print()
    print("Résultats retrieval :")

    for rank, item in enumerate(
        result["results"],
        start=1,
    ):
        print()
        print(
            f"#{rank} "
            f"| distance={item.distance:.4f} "
            f"| chunk={item.chunk_id}"
        )
        print(item.content)

    print()
    print("Réponse retrieval-only :")
    print(result["answer"])

    # ---------------------------------------------------------
    # Nettoyage des données Q&A uniquement.
    # Le document, ses chunks et embeddings sont conservés.
    # ---------------------------------------------------------

    question_id = result["question_id"]
    attempt_id = result["attempt_id"]

    engine = create_database_engine()

    with Session(engine) as session:
        session.exec(
            delete(QuestionFeedback).where(
                QuestionFeedback.question_id
                == question_id
            )
        )

        session.exec(
            delete(QuestionEvidence).where(
                QuestionEvidence.question_attempt_id
                == attempt_id
            )
        )

        session.exec(
            delete(QuestionAttempt).where(
                QuestionAttempt.question_id
                == question_id
            )
        )

        session.exec(
            delete(Question).where(
                Question.id == question_id
            )
        )

        session.commit()

    print()
    print("Données Q&A de test supprimées.")
    print("Document indexé conservé.")


if __name__ == "__main__":
    main()
