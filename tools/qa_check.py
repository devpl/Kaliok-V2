from sqlmodel import Session, delete

from kaliok.qa.service import (
    add_evidence,
    complete_attempt,
    create_attempt,
    create_question,
    get_attempts,
    get_question,
)
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    Question,
    QuestionAttempt,
    QuestionEvidence,
)


def main() -> None:
    question = create_question(
        "Quel est le prix TTC ?",
        origin="system",
    )

    print(
        f"Question créée : {question.id}"
    )

    attempt = create_attempt(
        question.id,
        strategy="vector",
        pipeline_version="qa-check-1",
    )

    print(
        f"Tentative créée : {attempt.id}"
    )

    evidence = add_evidence(
        attempt.id,
        rank=1,
        score=0.4714,
        evidence_text=(
            "PRIX T.T.C livrée et posée "
            "23 446,00 €"
        ),
    )

    print(
        f"Preuve créée : {evidence.id}"
    )

    complete_attempt(
        attempt.id,
        answer_text="23 446,00 €",
        confidence=0.95,
        resolved=True,
        resolution_reason="answer_supported_by_evidence",
    )

    stored_question = get_question(
        question.id
    )

    attempts = get_attempts(
        question.id
    )

    print()
    print("État final :")
    print(
        f"  question.status = "
        f"{stored_question.status}"
    )
    print(
        f"  attempts = {len(attempts)}"
    )
    print(
        f"  answer = {attempts[-1].answer_text}"
    )
    print(
        f"  attempt.status = "
        f"{attempts[-1].status}"
    )

    engine = create_database_engine()

    with Session(engine) as session:
        session.exec(
            delete(QuestionEvidence).where(
                QuestionEvidence.question_attempt_id
                == attempt.id
            )
        )

        session.exec(
            delete(QuestionAttempt).where(
                QuestionAttempt.question_id
                == question.id
            )
        )

        session.exec(
            delete(Question).where(
                Question.id == question.id
            )
        )

        session.commit()

    print()
    print("Nettoyage terminé.")


if __name__ == "__main__":
    main()
