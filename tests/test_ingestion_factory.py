from kaliok.ingestion_runtime.factory import create_ingestion_orchestrator
from kaliok.ingestion.orchestrator import IngestionOrchestrator


class FakeSession:
    pass


def test_create_ingestion_orchestrator():
    orchestrator = create_ingestion_orchestrator(FakeSession())

    assert isinstance(orchestrator, IngestionOrchestrator)