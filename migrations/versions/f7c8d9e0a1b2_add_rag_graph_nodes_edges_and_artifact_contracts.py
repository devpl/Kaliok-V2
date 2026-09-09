"""Add the descriptive RAG graph and artifact contracts.

The existing pipeline catalogue remains the runtime source of truth.  This
revision only adds a graph-shaped descriptive projection and preserves the
legacy JSONB artifact declarations and template relations.
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import uuid4

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f7c8d9e0a1b2"
down_revision: str | None = "c7d8e9f0a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB()


ARTIFACT_STORAGE = {
    "raw_document": ("document", "kaliok.document_version.v1"),
    "content_blocks": ("relational_table", "kaliok.content_blocks.v1"),
    "normalized_content_units": (
        "relational_table",
        "kaliok.normalized_content_units.v1",
    ),
    "discovered_candidates": (
        "relational_table",
        "kaliok.discovered_candidates.v1",
    ),
    "entities": ("relational_table", "kaliok.entities.v1"),
    "document_chunks": ("relational_table", "kaliok.document_chunks.v1"),
    "search_index": ("index", "kaliok.search_index.v1"),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _configuration(value: object, *, context: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Configuration JSON invalide pour {context}.")
    return value


def _artifact_values(value: object, *, context: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(
            f"Liste d'artefacts historique invalide pour {context} : attendu array de strings."
        )
    return value


def upgrade() -> None:
    op.create_table(
        "artifact_types",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type_key", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False, server_default=sa.text("'1'")),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("storage_kind", sa.String(), nullable=False),
        sa.Column("storage_reference", sa.String(), nullable=True),
        sa.Column("schema_definition", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("btrim(artifact_type_key) != ''", name="ck_artifact_types_key_nonempty"),
        sa.CheckConstraint("btrim(version) != ''", name="ck_artifact_types_version_nonempty"),
        sa.CheckConstraint("jsonb_typeof(schema_definition) = 'object'", name="ck_artifact_types_schema_definition_object"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_type_key", "version", name="uq_artifact_types_key_version"),
    )
    op.create_index(
        "ix_artifact_types_key_active",
        "artifact_types",
        ["artifact_type_key", "is_active"],
    )
    op.create_index("ix_artifact_types_storage_kind", "artifact_types", ["storage_kind"])

    op.create_table(
        "capability_artifact_contracts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("capability_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type_id", sa.Uuid(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("port_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("required", sa.Boolean(), nullable=True),
        sa.Column("cardinality", sa.String(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("direction IN ('input', 'output')", name="ck_capability_artifact_contracts_direction"),
        sa.CheckConstraint("position >= 0", name="ck_capability_artifact_contracts_position_nonnegative"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_capability_artifact_contracts_configuration_object"),
        sa.ForeignKeyConstraint(["capability_id"], ["capabilities.id"]),
        sa.ForeignKeyConstraint(["artifact_type_id"], ["artifact_types.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "capability_id",
            "direction",
            "port_key",
            name="uq_capability_artifact_contracts_capability_direction_port",
        ),
    )
    op.create_index(
        "ix_capability_artifact_contracts_capability_id",
        "capability_artifact_contracts",
        ["capability_id"],
    )
    op.create_index(
        "ix_capability_artifact_contracts_artifact_type_id",
        "capability_artifact_contracts",
        ["artifact_type_id"],
    )

    op.create_table(
        "rag_template_nodes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("node_key", sa.String(), nullable=False),
        sa.Column("capability_id", sa.Uuid(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("zone_key", sa.String(), nullable=True),
        sa.Column("requirement_mode", sa.String(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("btrim(node_key) != ''", name="ck_rag_template_nodes_key_nonempty"),
        sa.CheckConstraint("btrim(display_name) != ''", name="ck_rag_template_nodes_display_name_nonempty"),
        sa.CheckConstraint("position >= 0", name="ck_rag_template_nodes_position_nonnegative"),
        sa.CheckConstraint("requirement_mode IN ('required', 'optional')", name="ck_rag_template_nodes_requirement_mode"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_rag_template_nodes_configuration_object"),
        sa.ForeignKeyConstraint(["rag_template_revision_id"], ["rag_template_revisions.id"]),
        sa.ForeignKeyConstraint(["capability_id"], ["capabilities.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rag_template_revision_id", "node_key", name="uq_rag_template_nodes_revision_key"),
        sa.UniqueConstraint("id", "rag_template_revision_id", name="uq_rag_template_nodes_id_revision"),
    )
    op.create_index("ix_rag_template_nodes_revision_id", "rag_template_nodes", ["rag_template_revision_id"])
    op.create_index("ix_rag_template_nodes_capability_id", "rag_template_nodes", ["capability_id"])

    op.create_table(
        "rag_template_edges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("source_node_id", sa.Uuid(), nullable=False),
        sa.Column("target_node_id", sa.Uuid(), nullable=False),
        sa.Column("edge_key", sa.String(), nullable=False),
        sa.Column("edge_type", sa.String(), nullable=False, server_default=sa.text("'normal'")),
        sa.Column("source_port_key", sa.String(), nullable=True),
        sa.Column("target_port_key", sa.String(), nullable=True),
        sa.Column("condition", JSONB, nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("source_node_id != target_node_id", name="ck_rag_template_edges_distinct_nodes"),
        sa.CheckConstraint("btrim(edge_key) != ''", name="ck_rag_template_edges_key_nonempty"),
        sa.CheckConstraint("btrim(edge_type) != ''", name="ck_rag_template_edges_type_nonempty"),
        sa.CheckConstraint("priority >= 0", name="ck_rag_template_edges_priority_nonnegative"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_rag_template_edges_configuration_object"),
        sa.CheckConstraint("condition IS NULL OR jsonb_typeof(condition) = 'object'", name="ck_rag_template_edges_condition_object"),
        sa.ForeignKeyConstraint(["rag_template_revision_id"], ["rag_template_revisions.id"]),
        sa.ForeignKeyConstraint(
            ["source_node_id", "rag_template_revision_id"],
            ["rag_template_nodes.id", "rag_template_nodes.rag_template_revision_id"],
            name="fk_rag_template_edges_source_node_revision",
        ),
        sa.ForeignKeyConstraint(
            ["target_node_id", "rag_template_revision_id"],
            ["rag_template_nodes.id", "rag_template_nodes.rag_template_revision_id"],
            name="fk_rag_template_edges_target_node_revision",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rag_template_revision_id", "edge_key", name="uq_rag_template_edges_revision_key"),
    )
    op.create_index("ix_rag_template_edges_revision_id", "rag_template_edges", ["rag_template_revision_id"])
    op.create_index("ix_rag_template_edges_source_node_id", "rag_template_edges", ["source_node_id"])
    op.create_index("ix_rag_template_edges_target_node_id", "rag_template_edges", ["target_node_id"])
    op.create_index("ix_rag_template_edges_revision_type", "rag_template_edges", ["rag_template_revision_id", "edge_type"])

    connection = op.get_bind()
    if context.is_offline_mode():
        return
    artifact_table = sa.table(
        "artifact_types",
        sa.column("id", sa.Uuid()),
        sa.column("artifact_type_key", sa.String()),
        sa.column("version", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("storage_kind", sa.String()),
        sa.column("storage_reference", sa.String()),
        sa.column("schema_definition", JSONB),
        sa.column("is_active", sa.Boolean()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    contract_table = sa.table(
        "capability_artifact_contracts",
        sa.column("id", sa.Uuid()),
        sa.column("capability_id", sa.Uuid()),
        sa.column("artifact_type_id", sa.Uuid()),
        sa.column("direction", sa.String()),
        sa.column("port_key", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("required", sa.Boolean()),
        sa.column("cardinality", sa.String()),
        sa.column("position", sa.Integer()),
        sa.column("configuration", JSONB),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    node_table = sa.table(
        "rag_template_nodes",
        sa.column("id", sa.Uuid()),
        sa.column("rag_template_revision_id", sa.Uuid()),
        sa.column("node_key", sa.String()),
        sa.column("capability_id", sa.Uuid()),
        sa.column("display_name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("zone_key", sa.String()),
        sa.column("requirement_mode", sa.String()),
        sa.column("position", sa.Integer()),
        sa.column("enabled", sa.Boolean()),
        sa.column("configuration", JSONB),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    edge_table = sa.table(
        "rag_template_edges",
        sa.column("id", sa.Uuid()),
        sa.column("rag_template_revision_id", sa.Uuid()),
        sa.column("source_node_id", sa.Uuid()),
        sa.column("target_node_id", sa.Uuid()),
        sa.column("edge_key", sa.String()),
        sa.column("edge_type", sa.String()),
        sa.column("source_port_key", sa.String()),
        sa.column("target_port_key", sa.String()),
        sa.column("condition", JSONB),
        sa.column("priority", sa.Integer()),
        sa.column("enabled", sa.Boolean()),
        sa.column("configuration", JSONB),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )

    capabilities = connection.execute(
        sa.text(
            "SELECT id, capability_key, display_name, description, "
            "input_artifact_types, output_artifact_types "
            "FROM capabilities ORDER BY display_order, capability_key, id"
        )
    ).mappings().all()
    artifact_ids = {
        row.artifact_type_key: row.id
        for row in connection.execute(
            sa.select(artifact_table.c.artifact_type_key, artifact_table.c.id).where(
                artifact_table.c.version == "1"
            )
        )
    }
    all_artifact_keys: list[str] = []
    seen_artifact_keys: set[str] = set()
    for capability in capabilities:
        for direction, values in (
            ("input", capability.input_artifact_types),
            ("output", capability.output_artifact_types),
        ):
            for artifact_key in _artifact_values(
                values,
                context=f"{capability.capability_key}.{direction}",
            ):
                if not artifact_key.strip():
                    continue
                if artifact_key in seen_artifact_keys:
                    continue
                if artifact_key not in ARTIFACT_STORAGE:
                    raise RuntimeError(
                        f"Artifact historique non qualifiable sans invention : {artifact_key!r}."
                    )
                seen_artifact_keys.add(artifact_key)
                all_artifact_keys.append(artifact_key)

    for artifact_key in all_artifact_keys:
        if artifact_key in artifact_ids:
            continue
        storage_kind, storage_reference = ARTIFACT_STORAGE[artifact_key]
        artifact_id = uuid4()
        now = _now()
        connection.execute(
            artifact_table.insert().values(
                id=artifact_id,
                artifact_type_key=artifact_key,
                version="1",
                display_name=artifact_key.replace("_", " ").strip().title(),
                description=None,
                storage_kind=storage_kind,
                storage_reference=storage_reference,
                schema_definition={},
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        )
        artifact_ids[artifact_key] = artifact_id

    for capability in capabilities:
        for direction, values in (
            ("input", capability.input_artifact_types),
            ("output", capability.output_artifact_types),
        ):
            values = _artifact_values(
                values,
                context=f"{capability.capability_key}.{direction}",
            )
            occurrences: defaultdict[str, int] = defaultdict(int)
            for position, artifact_key in enumerate(values):
                if not artifact_key.strip():
                    continue
                occurrences[artifact_key] += 1
                suffix = f"__{occurrences[artifact_key]}" if occurrences[artifact_key] > 1 else ""
                connection.execute(
                    contract_table.insert().values(
                        id=uuid4(),
                        capability_id=capability.id,
                        artifact_type_id=artifact_ids[artifact_key],
                        direction=direction,
                        port_key=f"{'in' if direction == 'input' else 'out'}_{artifact_key}{suffix}",
                        display_name=None,
                        required=None,
                        cardinality=None,
                        position=position,
                        configuration={},
                        created_at=_now(),
                    )
                )

    legacy_nodes = connection.execute(
        sa.text(
            "SELECT rtc.rag_template_revision_id, rtc.capability_id, "
            "c.capability_key, c.display_name, c.description, rtc.requirement_mode, "
            "rtc.display_order, rtc.configuration "
            "FROM rag_template_capabilities rtc "
            "JOIN capabilities c ON c.id = rtc.capability_id "
            "ORDER BY rtc.rag_template_revision_id, rtc.display_order, c.capability_key, rtc.id"
        )
    ).mappings().all()
    nodes_by_revision_capability: defaultdict[tuple[object, object], list[dict]] = defaultdict(list)
    node_keys: set[tuple[object, str]] = set()
    occurrences_by_revision_key: defaultdict[tuple[object, str], int] = defaultdict(int)
    for legacy in legacy_nodes:
        revision_capability = (legacy.rag_template_revision_id, legacy.capability_key)
        occurrences_by_revision_key[revision_capability] += 1
        node_key = f"{legacy.capability_key}__{occurrences_by_revision_key[revision_capability]}"
        if (legacy.rag_template_revision_id, node_key) in node_keys:
            raise RuntimeError(f"Collision de node_key pendant le backfill : {node_key}.")
        node_keys.add((legacy.rag_template_revision_id, node_key))
        node = {
            "id": uuid4(),
            "rag_template_revision_id": legacy.rag_template_revision_id,
            "node_key": node_key,
            "capability_id": legacy.capability_id,
            "display_name": legacy.display_name,
            "description": legacy.description,
            "zone_key": None,
            "requirement_mode": legacy.requirement_mode,
            "position": legacy.display_order,
            "enabled": True,
            "configuration": _configuration(
                legacy.configuration,
                context=f"node {node_key}",
            ),
            "created_at": _now(),
        }
        connection.execute(node_table.insert().values(**node))
        nodes_by_revision_capability[(legacy.rag_template_revision_id, legacy.capability_id)].append(node)

    dependencies = connection.execute(
        sa.text(
            "SELECT rag_template_revision_id, source_capability_id, target_capability_id "
            "FROM rag_template_dependencies ORDER BY rag_template_revision_id, id"
        )
    ).mappings().all()
    edge_keys: set[tuple[object, str]] = set()
    for dependency in dependencies:
        source_nodes = nodes_by_revision_capability[
            (dependency.rag_template_revision_id, dependency.source_capability_id)
        ]
        target_nodes = nodes_by_revision_capability[
            (dependency.rag_template_revision_id, dependency.target_capability_id)
        ]
        if len(source_nodes) != 1 or len(target_nodes) != 1:
            raise RuntimeError(
                "Dépendance ambiguë ou sans node pendant le backfill : "
                f"{dependency.rag_template_revision_id}/{dependency.source_capability_id}"
                f" -> {dependency.target_capability_id}."
            )
        source = source_nodes[0]
        target = target_nodes[0]
        edge_key = f"edge__{source['node_key']}__to__{target['node_key']}"
        if (dependency.rag_template_revision_id, edge_key) in edge_keys:
            raise RuntimeError(f"Collision de edge_key pendant le backfill : {edge_key}.")
        edge_keys.add((dependency.rag_template_revision_id, edge_key))
        connection.execute(
            edge_table.insert().values(
                id=uuid4(),
                rag_template_revision_id=dependency.rag_template_revision_id,
                source_node_id=source["id"],
                target_node_id=target["id"],
                edge_key=edge_key,
                edge_type="normal",
                source_port_key=None,
                target_port_key=None,
                condition=sa.null(),
                priority=0,
                enabled=True,
                configuration={},
                created_at=_now(),
            )
        )


def downgrade() -> None:
    op.drop_table("rag_template_edges")
    op.drop_table("rag_template_nodes")
    op.drop_table("capability_artifact_contracts")
    op.drop_table("artifact_types")
