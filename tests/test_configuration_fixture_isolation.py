from sqlalchemy import text
from sqlmodel import Session, select

from kaliok.configuration import bootstrap_rag_configuration
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import ConfigurationProfileRevision


def _public_referenced_revision_count() -> int:
    engine = create_database_engine()
    with engine.connect() as connection:
        return connection.execute(text(
            "SELECT count(DISTINCT revision.id) "
            "FROM public.configuration_profile_revisions AS revision "
            "JOIN public.question_attempts AS attempt "
            "ON attempt.configuration_revision_id = revision.id"
        )).scalar_one()


def test_configuration_fixture_preserves_referenced_public_revisions(
    configuration_session: Session,
) -> None:
    before = _public_referenced_revision_count()
    schema = configuration_session.exec(text("SELECT current_schema()")).one()[0]

    created = bootstrap_rag_configuration(configuration_session)
    isolated_revisions = configuration_session.exec(
        select(ConfigurationProfileRevision)
    ).all()

    assert schema.startswith("kaliok_test_")
    assert [revision.id for revision in isolated_revisions] == [created.revision_id]
    assert _public_referenced_revision_count() == before
