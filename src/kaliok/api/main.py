from fastapi import FastAPI

from kaliok.api.documents import router as documents_router


app = FastAPI(
    title="kaliok API",
    version="0.1.0",
)

app.include_router(documents_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "kaliok-api",
    }