from django.urls import path

from .views import (
    document_detail,
    home,
    rag_laboratory,
    rag_laboratory_data,
    rag_laboratory_pipeline,
    upload_document,
)
from .rag_prototype import rag_prototype


urlpatterns = [
    path("", home, name="home"),
    path("rag/prototype/", rag_prototype, name="rag_prototype"),
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
