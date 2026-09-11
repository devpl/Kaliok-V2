from fastapi import FastAPI

from kaliok.api.composer import router as composer_router
from kaliok.api.documents import router as documents_router
from kaliok.api.ingestion import router as ingestion_router
from kaliok.api.rag import router as rag_router
from kaliok.api.evaluation import router as evaluation_router

app = FastAPI(
    title="kaliok API",
    version="0.1.0",
)

app.include_router(documents_router)
app.include_router(ingestion_router)
app.include_router(rag_router)
app.include_router(evaluation_router)
app.include_router(composer_router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "kaliok-api",
    }
