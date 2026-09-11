from fastapi import APIRouter, Depends
from sqlmodel import Session

from kaliok.api.dependencies import get_session
from kaliok.pipeline.composer import RagComposerReadService


router = APIRouter(prefix="/rag/composer", tags=["rag-composer"])


@router.get("")
def get_rag_composer(session: Session = Depends(get_session)):
    """Return the persisted, read-only RAG composition catalogue."""
    return RagComposerReadService(session).project()
