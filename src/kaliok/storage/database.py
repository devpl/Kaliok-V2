import os

from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlmodel import create_engine

from kaliok.paths import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


def get_database_url() -> str:
    return (
        f"postgresql+psycopg://"
        f"{os.environ['KALIOK_DB_USER']}:"
        f"{os.environ['KALIOK_DB_PASSWORD']}@"
        f"{os.environ['KALIOK_DB_HOST']}:"
        f"{os.environ['KALIOK_DB_PORT']}/"
        f"{os.environ['KALIOK_DB_NAME']}"
    )


def create_database_engine() -> Engine:
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )