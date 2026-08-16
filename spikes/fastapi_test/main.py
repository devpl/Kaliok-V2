import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
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

app = FastAPI()


@app.get("/")
def root():
    return {"message": "kaliok V2 - FastAPI OK"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-health")
def db_health():
    with psycopg.connect(
        host=os.environ["KALIOK_DB_HOST"],
        port=os.environ["KALIOK_DB_PORT"],
        dbname=os.environ["KALIOK_DB_NAME"],
        user=os.environ["KALIOK_DB_USER"],
        password=os.environ["KALIOK_DB_PASSWORD"],
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_database(), current_user, extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
            )
            database, user, vector_version = cursor.fetchone()

    return {
        "status": "ok",
        "database": database,
        "user": user,
        "pgvector": vector_version,
    }


@app.post("/documents")
def create_document(document: DocumentTest):
    with Session(engine) as session:
        session.add(document)
        session.commit()
        session.refresh(document)
        return document


@app.get("/documents")
def list_documents():
    with Session(engine) as session:
        return session.exec(select(DocumentTest)).all()
