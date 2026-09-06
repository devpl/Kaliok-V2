"""add rag evaluation lab models

Revision ID: b71e3c9f2a10
Revises: c4e8a12d9f33
Create Date: 2026-09-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "b71e3c9f2a10"
down_revision: Union[str, Sequence[str], None] = "c4e8a12d9f33"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "questions",
        sa.Column("expected_answer", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_questions_document_id",
        "questions",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_questions_document_version_id",
        "questions",
        ["document_version_id"],
        unique=False,
    )

    op.create_table(
        "evaluation_suites",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "evaluation_suite_questions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("evaluation_suite_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["evaluation_suite_id"],
            ["evaluation_suites.id"],
        ),
        sa.ForeignKeyConstraint(
            ["question_id"],
            ["questions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "evaluation_suite_id",
            "question_id",
            name="uq_evaluation_suite_questions_question",
        ),
        sa.UniqueConstraint(
            "evaluation_suite_id",
            "position",
            name="uq_evaluation_suite_questions_position",
        ),
    )
    op.create_index(
        "ix_evaluation_suite_questions_evaluation_suite_id",
        "evaluation_suite_questions",
        ["evaluation_suite_id"],
        unique=False,
    )
    op.create_index(
        "ix_evaluation_suite_questions_question_id",
        "evaluation_suite_questions",
        ["question_id"],
        unique=False,
    )

    op.create_table(
        "evaluation_campaigns",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("suite_id", sa.Uuid(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "configuration",
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["suite_id"],
            ["evaluation_suites.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column(
        "question_attempts",
        sa.Column("configuration_revision_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "question_attempts",
        sa.Column("evaluation_campaign_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_question_attempts_configuration_revision_id",
        "question_attempts",
        "configuration_profile_revisions",
        ["configuration_revision_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_question_attempts_evaluation_campaign_id",
        "question_attempts",
        "evaluation_campaigns",
        ["evaluation_campaign_id"],
        ["id"],
    )
    op.create_unique_constraint(
        "uq_question_attempts_question_number",
        "question_attempts",
        ["question_id", "attempt_number"],
    )
    op.create_index(
        "ix_question_attempts_question_id",
        "question_attempts",
        ["question_id"],
        unique=False,
    )
    op.create_index(
        "ix_question_attempts_configuration_revision_id",
        "question_attempts",
        ["configuration_revision_id"],
        unique=False,
    )
    op.create_index(
        "ix_question_attempts_evaluation_campaign_id",
        "question_attempts",
        ["evaluation_campaign_id"],
        unique=False,
    )

    op.create_index(
        "ix_question_evidence_question_attempt_id",
        "question_evidence",
        ["question_attempt_id"],
        unique=False,
    )
    op.create_index(
        "ix_question_feedback_question_id",
        "question_feedback",
        ["question_id"],
        unique=False,
    )
    op.create_index(
        "ix_question_feedback_question_attempt_id",
        "question_feedback",
        ["question_attempt_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_question_feedback_question_attempt_id",
        table_name="question_feedback",
    )
    op.drop_index(
        "ix_question_feedback_question_id",
        table_name="question_feedback",
    )
    op.drop_index(
        "ix_question_evidence_question_attempt_id",
        table_name="question_evidence",
    )

    op.drop_index(
        "ix_question_attempts_evaluation_campaign_id",
        table_name="question_attempts",
    )
    op.drop_index(
        "ix_question_attempts_configuration_revision_id",
        table_name="question_attempts",
    )
    op.drop_index(
        "ix_question_attempts_question_id",
        table_name="question_attempts",
    )
    op.drop_constraint(
        "uq_question_attempts_question_number",
        "question_attempts",
        type_="unique",
    )
    op.drop_constraint(
        "fk_question_attempts_evaluation_campaign_id",
        "question_attempts",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_question_attempts_configuration_revision_id",
        "question_attempts",
        type_="foreignkey",
    )
    op.drop_column("question_attempts", "evaluation_campaign_id")
    op.drop_column("question_attempts", "configuration_revision_id")

    op.drop_table("evaluation_campaigns")

    op.drop_index(
        "ix_evaluation_suite_questions_question_id",
        table_name="evaluation_suite_questions",
    )
    op.drop_index(
        "ix_evaluation_suite_questions_evaluation_suite_id",
        table_name="evaluation_suite_questions",
    )
    op.drop_table("evaluation_suite_questions")
    op.drop_table("evaluation_suites")

    op.drop_index(
        "ix_questions_document_version_id",
        table_name="questions",
    )
    op.drop_index(
        "ix_questions_document_id",
        table_name="questions",
    )
    op.drop_column("questions", "expected_answer")
