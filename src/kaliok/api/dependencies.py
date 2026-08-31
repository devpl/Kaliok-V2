from collections.abc import Generator

from sqlmodel import Session

from kaliok.storage.database import create_database_engine


engine = create_database_engine()


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session