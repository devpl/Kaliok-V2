from django.urls import path

from .views import document_detail, home, rag_laboratory, rag_laboratory_data, upload_document


urlpatterns = [
    path("", home, name="home"),
    path("rag/laboratory/", rag_laboratory, name="rag_laboratory"),
    path("rag/laboratory/data/", rag_laboratory_data, name="rag_laboratory_data"),
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
