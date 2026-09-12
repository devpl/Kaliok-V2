from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from kaliok.api.dependencies import get_session
from kaliok.pipeline.composer import RagComposerReadService
from kaliok.pipeline.composer_mutations import RagComposerMutationService
from kaliok.pipeline.step_testing import RagComposerStepTestService, StepInput, StepTestRefused


router = APIRouter(prefix="/rag/composer", tags=["rag-composer"])


@router.get("")
def get_rag_composer(session: Session = Depends(get_session)):
    """Return the persisted, read-only RAG composition catalogue."""
    return RagComposerReadService(session).project()


class ComposerForkRequest(BaseModel):
    pipeline_revision_id: UUID


class ComposerCreateNodeRequest(BaseModel):
    pipeline_revision_id: UUID
    capability_id: UUID


class ComposerSelectToolRequest(BaseModel):
    pipeline_revision_id: UUID
    rag_template_node_id: UUID
    component_version_id: UUID
    configuration: dict = Field(default_factory=dict)


class ComposerCreateEdgeRequest(BaseModel):
    pipeline_revision_id: UUID
    source_node_id: UUID
    target_node_id: UUID
    source_port_key: str | None = None
    target_port_key: str | None = None


def _mutation_error(error: ValueError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))


@router.post("/revisions/fork", status_code=status.HTTP_201_CREATED)
def fork_rag_composer_revision(
    request: ComposerForkRequest, session: Session = Depends(get_session)
):
    try:
        revision = RagComposerMutationService(session).fork_pipeline_revision_with_graph(
            request.pipeline_revision_id
        )
        session.commit()
        return {
            "status": "created",
            "pipeline_revision_id": str(revision.id),
            "rag_template_revision_id": str(revision.rag_template_revision_id),
            "revision_number": revision.revision_number,
        }
    except ValueError as error:
        session.rollback()
        raise _mutation_error(error) from error


@router.post("/nodes", status_code=status.HTTP_201_CREATED)
def create_rag_composer_node(
    request: ComposerCreateNodeRequest, session: Session = Depends(get_session)
):
    try:
        node = RagComposerMutationService(session).create_node(
            request.pipeline_revision_id, request.capability_id
        )
        session.commit()
        return {
            "status": "created",
            "pipeline_revision_id": str(request.pipeline_revision_id),
            "rag_template_node_id": str(node.id),
            "node_key": node.node_key,
        }
    except ValueError as error:
        session.rollback()
        raise _mutation_error(error) from error


@router.post("/revisions/empty", status_code=status.HTTP_201_CREATED)
def create_empty_rag_composer_revision(request: ComposerForkRequest, session: Session = Depends(get_session)):
    try:
        revision = RagComposerMutationService(session).create_empty_revision(request.pipeline_revision_id)
        session.commit()
        return {"status": "created", "pipeline_revision_id": str(revision.id),
                "rag_template_revision_id": str(revision.rag_template_revision_id),
                "revision_number": revision.revision_number}
    except ValueError as error:
        session.rollback()
        raise _mutation_error(error) from error


@router.post("/nodes/select-tool")
def select_rag_composer_node_tool(request: ComposerSelectToolRequest, session: Session = Depends(get_session)):
    try:
        link = RagComposerMutationService(session).select_node_tool(
            request.pipeline_revision_id, request.rag_template_node_id,
            request.component_version_id, configuration=request.configuration,
        )
        session.commit()
        return {"status": "configured", "pipeline_revision_id": str(request.pipeline_revision_id),
                "rag_template_node_id": str(request.rag_template_node_id),
                "pipeline_binding_id": str(link.pipeline_binding_id)}
    except ValueError as error:
        session.rollback()
        raise _mutation_error(error) from error


@router.post("/edges", status_code=status.HTTP_201_CREATED)
def create_rag_composer_edge(request: ComposerCreateEdgeRequest, session: Session = Depends(get_session)):
    try:
        edge = RagComposerMutationService(session).create_edge(
            request.pipeline_revision_id, request.source_node_id, request.target_node_id,
            source_port_key=request.source_port_key, target_port_key=request.target_port_key,
        )
        session.commit()
        return {"status": "created", "pipeline_revision_id": str(request.pipeline_revision_id),
                "rag_template_edge_id": str(edge.id)}
    except ValueError as error:
        session.rollback()
        raise _mutation_error(error) from error


class ComposerStepInputRequest(BaseModel):
    artifact_type_key: str
    artifact_id: UUID
    port_key: str | None = None


class ComposerStepRequest(BaseModel):
    pipeline_revision_id: UUID
    rag_template_node_id: UUID
    document_version_id: UUID | None = None
    inputs: list[ComposerStepInputRequest] = Field(default_factory=list)


@router.post("/execute-step")
def execute_rag_composer_step(request: ComposerStepRequest, session: Session = Depends(get_session)):
    """Run one selected node only; it never resolves graph prerequisites."""
    try:
        return RagComposerStepTestService(session).execute(
            pipeline_revision_id=request.pipeline_revision_id,
            node_id=request.rag_template_node_id,
            document_version_id=request.document_version_id,
            inputs=[StepInput(**item.model_dump()) for item in request.inputs],
        )
    except StepTestRefused as error:
        return {
            "status": "refused",
            "error": str(error) if error.kind not in {"missing_inputs", "unsupported"} else None,
            "missing_inputs": error.details if error.kind == "missing_inputs" else [],
            "unsupported_reason": str(error) if error.kind == "unsupported" else None,
        }
