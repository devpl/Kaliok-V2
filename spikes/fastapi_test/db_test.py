import os
from pathlib import Path

from dotenv import load_dotenv
from sqlmodel import Field, Session, SQLModel, create_engine, select

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE)

DATABASE_URL = (
    f"postgresql+psycopg://"
    f"{os.environ['KALIOK_DB_USER']}:"
    f"{os.environ['KALIOK_DB_PASSWORD']}@"
    f"{os.environ['KALIOK_DB_HOST']}:"
    f"{os.environ['KALIOK_DB_PORT']}/"
    f"{os.environ['KALIOK_DB_NAME']}"
)

engine = create_engine(DATABASE_URL)


class DocumentTest(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str


SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    document = DocumentTest(title="Premier document kaliok")
    session.add(document)
    session.commit()
    session.refresh(document)

    results = session.exec(select(DocumentTest)).all()

    for item in results:
        print(item)
