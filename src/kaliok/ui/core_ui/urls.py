from django.urls import path

from .views import document_detail, home


urlpatterns = [
    path("", home, name="home"),
    path(
        "documents/<uuid:document_id>/",
        document_detail,
        name="document_detail",
    ),
]