from uuid import uuid4

from kaliok.storage.models import ProcessingRun


def test_processing_run_allows_documentless_future_multi_document_scope():
    run = ProcessingRun(process_type="entity_resolution", status="running")
    assert run.document_version_id is None


def test_existing_processing_run_contract_still_accepts_document_version():
    document_version_id = uuid4()
    run = ProcessingRun(
        document_version_id=document_version_id,
        process_type="candidate_discovery",
        status="completed",
    )
    assert run.document_version_id == document_version_id
