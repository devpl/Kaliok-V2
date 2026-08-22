from fastapi import FastAPI
from sqlalchemy import text

from kaliok.storage.database import create_database_engine


app = FastAPI(
    title="kaliok V2",
    version="0.1.0",
)

engine = create_database_engine()


@app.get("/")
def root():
    return {"message": "kaliok V2 - FastAPI OK"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-health")
def db_health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "database": "connected",
        }

    except Exception as exc:
        return {
            "status": "error",
            "database": "unavailable",
            "detail": str(exc),
        }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8010,
        reload=True,
    )