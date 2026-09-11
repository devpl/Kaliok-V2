from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session

from kaliok.api.dependencies import get_session
from kaliok.pipeline.composer import RagComposerReadService
from kaliok.pipeline.step_testing import RagComposerStepTestService, StepInput, StepTestRefused


router = APIRouter(prefix="/rag/composer", tags=["rag-composer"])


@router.get("")
def get_rag_composer(session: Session = Depends(get_session)):
    """Return the persisted, read-only RAG composition catalogue."""
    return RagComposerReadService(session).project()


class ComposerStepInputRequest(BaseModel):
    artifact_type_key: str
    artifact_id: UUID
    port_key: str | None = None


class ComposerStepRequest(BaseModel):
    pipeline_revision_id: UUID
    rag_template_node_id: UUID
    inputs: list[ComposerStepInputRequest] = Field(default_factory=list)


@router.post("/execute-step")
def execute_rag_composer_step(request: ComposerStepRequest, session: Session = Depends(get_session)):
    """Run one selected node only; it never resolves graph prerequisites."""
    try:
        return RagComposerStepTestService(session).execute(
            pipeline_revision_id=request.pipeline_revision_id,
            node_id=request.rag_template_node_id,
            inputs=[StepInput(**item.dict()) for item in request.inputs],
        )
    except StepTestRefused as error:
        return {
            "status": "refused",
            "error": str(error) if error.kind == "invalid_request" else None,
            "missing_inputs": error.details if error.kind == "missing_inputs" else [],
            "unsupported_reason": str(error) if error.kind == "unsupported" else None,
        }
