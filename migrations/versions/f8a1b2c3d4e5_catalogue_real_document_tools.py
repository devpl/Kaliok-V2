"""Expose catalogued real document tools in the Composer.

Docling's version is intentionally read from completed ProcessingRun rows: the
application does not own a fixed Docling product version.
"""

from collections.abc import Sequence
from uuid import uuid4

from alembic import op
import sqlalchemy as sa


revision: str = "f8a1b2c3d4e5"
down_revision: str | None = "d3e4f5a6b7c8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


LABELS = {
    "kaliok-reader": "Lecture documentaire Kaliok",
    "kaliok-normalizer": "Normalisation Kaliok",
    "kaliok-candidate-discovery": "Découverte d’entités Kaliok",
    "kaliok-entity-resolution": "Résolution d’entités Kaliok",
    "kaliok-semantic-chunker": "Découpage sémantique Kaliok",
    "postgres-normalized-index": "Index PostgreSQL Kaliok",
}


def upgrade() -> None:
    bind = op.get_bind()
    for key, label in LABELS.items():
        bind.execute(
            sa.text("UPDATE components SET display_name = :label WHERE component_key = :key"),
            {"key": key, "label": label},
        )

    versions = bind.execute(sa.text("""
        SELECT DISTINCT engine_version
        FROM processing_runs
        WHERE engine = 'docling'
          AND status = 'completed'
          AND engine_version IS NOT NULL
        ORDER BY engine_version
    """)).scalars().all()
    if not versions:
        return

    component_id = bind.execute(
        sa.text("SELECT id FROM components WHERE component_key = 'docling'")
    ).scalar_one_or_none()
    if component_id is None:
        component_id = uuid4()
        bind.execute(sa.text("""
            INSERT INTO components (id, component_key, display_name, description, is_active, created_at, updated_at)
            VALUES (:id, 'docling', 'Docling', :description, true, now(), now())
        """), {"id": component_id, "description": (
            "Moteur Docling observé dans les traitements documentaires Kaliok. "
            "Les sorties conservées incluent la structure et les tables quand elles sont fournies par le document converti."
        )})
    else:
        bind.execute(sa.text("""
            UPDATE components SET display_name = 'Docling', description = :description
            WHERE id = :id
        """), {"id": component_id, "description": (
            "Moteur Docling observé dans les traitements documentaires Kaliok. "
            "Les sorties conservées incluent la structure et les tables quand elles sont fournies par le document converti."
        )})

    extraction_id = bind.execute(
        sa.text("SELECT id FROM capabilities WHERE capability_key = 'document_extraction'")
    ).scalar_one()
    for version in versions:
        version_id = bind.execute(sa.text("""
            SELECT id FROM component_versions
            WHERE component_id = :component_id AND version = :version
        """), {"component_id": component_id, "version": version}).scalar_one_or_none()
        if version_id is None:
            version_id = uuid4()
            bind.execute(sa.text("""
                INSERT INTO component_versions (id, component_id, version, status, configuration_schema, metadata, created_at)
                VALUES (:id, :component_id, :version, 'available', '{}'::jsonb,
                        '{"source":"processing_runs.engine_version","document_capabilities":["structure","tables"]}'::jsonb, now())
            """), {"id": version_id, "component_id": component_id, "version": version})
        exists = bind.execute(sa.text("""
            SELECT 1 FROM component_capabilities
            WHERE component_version_id = :version_id AND capability_id = :capability_id
        """), {"version_id": version_id, "capability_id": extraction_id}).scalar_one_or_none()
        if exists is None:
            bind.execute(sa.text("""
                INSERT INTO component_capabilities (id, component_version_id, capability_id, invocation_mode, configuration_schema, metadata, created_at)
                VALUES (:id, :version_id, :capability_id, 'independent', '{}'::jsonb,
                        '{"used_capability":"Lecture du document"}'::jsonb, now())
            """), {"id": uuid4(), "version_id": version_id, "capability_id": extraction_id})


def downgrade() -> None:
    # Catalogue names are descriptive data; do not erase a real tool or its
    # historical binding on downgrade.
    pass
