from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlmodel import Session, SQLModel

from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ConfigurationValue,
    ConfigurationValueOption,
    SettingCategory,
    SettingDefinition,
    SettingOption,
)


CONFIGURATION_TABLES = [
    SettingCategory.__table__,
    SettingDefinition.__table__,
    SettingOption.__table__,
    ConfigurationProfile.__table__,
    ConfigurationProfileRevision.__table__,
    ConfigurationValue.__table__,
    ConfigurationValueOption.__table__,
]


@pytest.fixture
def configuration_session() -> Iterator[Session]:
    """Give configuration tests empty, transaction-scoped PostgreSQL tables.

    The application database can contain profiles referenced by real attempts.
    A private schema avoids deleting or shadow-mutating any of those rows.
    """
    engine = create_database_engine()
    schema_name = f"kaliok_test_{uuid4().hex}"

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
            connection.execute(text(f'SET LOCAL search_path TO "{schema_name}"'))
            SQLModel.metadata.create_all(connection, tables=CONFIGURATION_TABLES)
            # This production invariant is migration-defined rather than part
            # of SQLModel metadata, so reproduce it in the isolated schema.
            connection.execute(text(
                "CREATE UNIQUE INDEX "
                "uq_configuration_profile_revisions_single_active "
                "ON configuration_profile_revisions (profile_id) "
                "WHERE status = 'active'"
            ))
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
