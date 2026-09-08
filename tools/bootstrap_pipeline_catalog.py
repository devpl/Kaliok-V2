"""Apply the qualified Kaliok descriptive catalogue after Alembic migration."""

from sqlmodel import Session

from kaliok.pipeline.persistence import bootstrap_catalog
from kaliok.storage.database import create_database_engine


def main() -> None:
    with Session(create_database_engine()) as session:
        result = bootstrap_catalog(session)
        session.commit()
    print({key: str(value) for key, value in result.items()})


if __name__ == "__main__":
    main()
