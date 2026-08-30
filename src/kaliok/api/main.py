from fastapi import FastAPI


app = FastAPI(
    title="kaliok API",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "kaliok-api",
    }