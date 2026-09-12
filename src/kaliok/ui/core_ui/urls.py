from django.urls import path

from .views import (
    document_detail,
    home,
    rag_laboratory,
    rag_laboratory_data,
    rag_laboratory_pipeline,
    upload_document,
)
from .rag_prototype import (
    composer_create_node,
    composer_create_empty_revision,
    composer_select_tool,
    composer_create_edge,
    composer_data,
    composer_fork_revision,
    composer_step_test,
    rag_prototype,
)


urlpatterns = [
    path("", home, name="home"),
    path("rag/prototype/", rag_prototype, name="rag_prototype"),
    path("rag/prototype/data/", composer_data, name="rag_prototype_data"),
    path("rag/prototype/revisions/fork/", composer_fork_revision, name="rag_prototype_fork"),
    path("rag/prototype/nodes/", composer_create_node, name="rag_prototype_create_node"),
    path("rag/prototype/revisions/empty/", composer_create_empty_revision, name="rag_prototype_empty"),
    path("rag/prototype/nodes/select-tool/", composer_select_tool, name="rag_prototype_select_tool"),
    path("rag/prototype/edges/", composer_create_edge, name="rag_prototype_create_edge"),
    path("rag/prototype/execute-step/", composer_step_test, name="rag_prototype_step_test"),
    path("rag/laboratory/", rag_laboratory, name="rag_laboratory"),
    path("rag/laboratory/data/", rag_laboratory_data, name="rag_laboratory_data"),
    path("rag/laboratory/pipeline/", rag_laboratory_pipeline, name="rag_laboratory_pipeline"),
    path(
        "documents/upload/",
        upload_document,
        name="upload_document",
    ),
    path(
        "documents/<uuid:document_id>/",
        document_detail,
        name="document_detail",
    ),
]
