"""add configuration profiles

Revision ID: f2b1d9c7a6e0
Revises: a84c7e2f19d1
Create Date: 2026-09-01

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "f2b1d9c7a6e0"
down_revision: str | None = "a84c7e2f19d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "setting_categories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("parent_category_id", sa.Uuid(), nullable=True),
        sa.Column("category_key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["parent_category_id"],
            ["setting_categories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "category_key",
            name="uq_setting_categories_category_key",
        ),
    )

    op.create_index(
        "ix_setting_categories_parent_category_id",
        "setting_categories",
        ["parent_category_id"],
        unique=False,
    )

    op.create_index(
        "ix_setting_categories_category_key",
        "setting_categories",
        ["category_key"],
        unique=False,
    )

    op.create_table(
        "setting_definitions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("category_id", sa.Uuid(), nullable=False),
        sa.Column("setting_key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("value_type", sa.String(), nullable=False),
        sa.Column("is_required", sa.Boolean(), nullable=False),
        sa.Column("is_editable", sa.Boolean(), nullable=False),
        sa.Column("is_encrypted", sa.Boolean(), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "value_type IN ("
            "'string', "
            "'integer', "
            "'float', "
            "'boolean', "
            "'choice', "
            "'multichoice'"
            ")",
            name="ck_setting_definitions_value_type",
        ),
        sa.ForeignKeyConstraint(
            ["category_id"],
            ["setting_categories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "category_id",
            "setting_key",
            name="uq_setting_definitions_category_key",
        ),
    )

    op.create_index(
        "ix_setting_definitions_category_id",
        "setting_definitions",
        ["category_id"],
        unique=False,
    )

    op.create_table(
        "setting_options",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "setting_definition_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column("option_key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["setting_definition_id"],
            ["setting_definitions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "setting_definition_id",
            "option_key",
            name="uq_setting_options_definition_key",
        ),
    )

    op.create_index(
        "ix_setting_options_setting_definition_id",
        "setting_options",
        ["setting_definition_id"],
        unique=False,
    )

    op.create_table(
        "configuration_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_key", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_key",
            name="uq_configuration_profiles_profile_key",
        ),
    )

    op.create_index(
        "ix_configuration_profiles_profile_key",
        "configuration_profiles",
        ["profile_key"],
        unique=False,
    )

    op.create_index(
        "uq_configuration_profiles_single_default",
        "configuration_profiles",
        ["is_default"],
        unique=True,
        postgresql_where=sa.text("is_default = true"),
    )

    op.create_table(
        "configuration_profile_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("change_reason", sa.String(), nullable=True),
        sa.Column(
            "created_by_actor_type",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "created_by_user_id",
            sa.String(),
            nullable=True,
        ),
        sa.Column(
            "created_by_display_name",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "activated_by_user_id",
            sa.String(),
            nullable=True,
        ),
        sa.Column(
            "activated_by_display_name",
            sa.String(),
            nullable=True,
        ),
        sa.Column(
            "activated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "revision_number > 0",
            name="ck_configuration_profile_revisions_number_positive",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'active', 'retired')",
            name="ck_configuration_profile_revisions_status",
        ),
        sa.CheckConstraint(
            "created_by_actor_type IN "
            "('user', 'system', 'service', 'migration')",
            name="ck_configuration_profile_revisions_actor_type",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["configuration_profiles.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id",
            "revision_number",
            name="uq_configuration_profile_revisions_number",
        ),
    )

    op.create_index(
        "ix_configuration_profile_revisions_profile_id",
        "configuration_profile_revisions",
        ["profile_id"],
        unique=False,
    )

    op.create_table(
        "configuration_values",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("revision_id", sa.Uuid(), nullable=False),
        sa.Column(
            "setting_definition_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column("value_text", sa.String(), nullable=True),
        sa.Column("value_integer", sa.Integer(), nullable=True),
        sa.Column("value_float", sa.Float(), nullable=True),
        sa.Column("value_boolean", sa.Boolean(), nullable=True),
        sa.Column("selected_option_id", sa.Uuid(), nullable=True),
        sa.Column("encrypted_value", sa.String(), nullable=True),
        sa.Column("encryption_key_id", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["configuration_profile_revisions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["setting_definition_id"],
            ["setting_definitions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["selected_option_id"],
            ["setting_options.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "revision_id",
            "setting_definition_id",
            name="uq_configuration_values_revision_setting",
        ),
    )

    op.create_index(
        "ix_configuration_values_revision_id",
        "configuration_values",
        ["revision_id"],
        unique=False,
    )

    op.create_index(
        "ix_configuration_values_setting_definition_id",
        "configuration_values",
        ["setting_definition_id"],
        unique=False,
    )

    op.create_table(
        "configuration_value_options",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "configuration_value_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "setting_option_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["configuration_value_id"],
            ["configuration_values.id"],
        ),
        sa.ForeignKeyConstraint(
            ["setting_option_id"],
            ["setting_options.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "configuration_value_id",
            "setting_option_id",
            name="uq_configuration_value_options_selection",
        ),
    )

    op.create_index(
        "ix_configuration_value_options_configuration_value_id",
        "configuration_value_options",
        ["configuration_value_id"],
        unique=False,
    )

    op.create_index(
        "ix_configuration_value_options_setting_option_id",
        "configuration_value_options",
        ["setting_option_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_configuration_value_options_setting_option_id",
        table_name="configuration_value_options",
    )

    op.drop_index(
        "ix_configuration_value_options_configuration_value_id",
        table_name="configuration_value_options",
    )

    op.drop_table("configuration_value_options")

    op.drop_index(
        "ix_configuration_values_setting_definition_id",
        table_name="configuration_values",
    )

    op.drop_index(
        "ix_configuration_values_revision_id",
        table_name="configuration_values",
    )

    op.drop_table("configuration_values")

    op.drop_index(
        "ix_configuration_profile_revisions_profile_id",
        table_name="configuration_profile_revisions",
    )

    op.drop_table("configuration_profile_revisions")

    op.drop_index(
        "uq_configuration_profiles_single_default",
        table_name="configuration_profiles",
    )

    op.drop_index(
        "ix_configuration_profiles_profile_key",
        table_name="configuration_profiles",
    )

    op.drop_table("configuration_profiles")

    op.drop_index(
        "ix_setting_options_setting_definition_id",
        table_name="setting_options",
    )

    op.drop_table("setting_options")

    op.drop_index(
        "ix_setting_definitions_category_id",
        table_name="setting_definitions",
    )

    op.drop_table("setting_definitions")

    op.drop_index(
        "ix_setting_categories_category_key",
        table_name="setting_categories",
    )

    op.drop_index(
        "ix_setting_categories_parent_category_id",
        table_name="setting_categories",
    )

    op.drop_table("setting_categories")