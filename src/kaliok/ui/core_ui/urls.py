from django.urls import path

from .views import document_detail, home, upload_document


urlpatterns = [
    path("", home, name="home"),
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