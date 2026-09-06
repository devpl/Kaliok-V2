from __future__ import annotations

import pytest
from sqlmodel import Session, select

from kaliok.configuration import bootstrap_rag_configuration
from kaliok.storage.models import (
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ConfigurationValue,
    SettingCategory,
    SettingDefinition,
    SettingOption,
)


def test_bootstrap_creates_rag_categories(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    categories = configuration_session.exec(
        select(SettingCategory)
    ).all()

    by_key = {
        category.category_key: category
        for category in categories
    }

    assert set(by_key) == {
        "rag",
        "rag.generation",
        "rag.retrieval",
        "rag.context",
    }

    assert by_key["rag"].parent_category_id is None
    assert by_key["rag.generation"].parent_category_id == by_key["rag"].id
    assert by_key["rag.retrieval"].parent_category_id == by_key["rag"].id
    assert by_key["rag.context"].parent_category_id == by_key["rag"].id


def test_bootstrap_creates_rag_setting_definitions(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    definitions = configuration_session.exec(
        select(SettingDefinition)
    ).all()

    categories = {
        category.id: category.category_key
        for category in configuration_session.exec(
            select(SettingCategory)
        ).all()
    }

    actual = {
        (
            categories[definition.category_id],
            definition.setting_key,
            definition.value_type,
        )
        for definition in definitions
    }

    assert actual == {
        ("rag.generation", "provider", "choice"),
        ("rag.generation", "model", "choice"),
        ("rag.generation", "temperature", "float"),
        ("rag.retrieval", "top_k", "integer"),
        ("rag.context", "builder", "choice"),
    }


def test_bootstrap_creates_generation_model_options(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    generation_category = configuration_session.exec(
        select(SettingCategory).where(
            SettingCategory.category_key == "rag.generation"
        )
    ).one()

    model_definition = configuration_session.exec(
        select(SettingDefinition).where(
            SettingDefinition.category_id == generation_category.id,
            SettingDefinition.setting_key == "model",
        )
    ).one()

    options = configuration_session.exec(
        select(SettingOption)
        .where(
            SettingOption.setting_definition_id
            == model_definition.id
        )
        .order_by(SettingOption.display_order)
    ).all()

    assert [option.option_key for option in options] == [
        "mistral",
        "gemma3:4b",
        "qwen3:8b",
        "phi4-mini",
    ]


def test_bootstrap_creates_default_profile_and_revision(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(
        configuration_session,
        actor_type="user",
        actor_user_id="user-42",
        actor_display_name="Alice Martin",
    )

    assert result.created is True

    profile = configuration_session.get(
        ConfigurationProfile,
        result.profile_id,
    )
    assert profile is not None

    assert profile.profile_key == "production-default"
    assert profile.is_active is True
    assert profile.is_default is True

    revision = configuration_session.get(
        ConfigurationProfileRevision,
        result.revision_id,
    )
    assert revision is not None

    assert revision.profile_id == profile.id
    assert revision.revision_number == 1
    assert revision.status == "active"

    assert revision.created_by_actor_type == "user"
    assert revision.created_by_user_id == "user-42"
    assert revision.created_by_display_name == "Alice Martin"

    assert revision.activated_by_user_id == "user-42"
    assert revision.activated_by_display_name == "Alice Martin"
    assert revision.activated_at is not None


def test_bootstrap_creates_expected_default_values(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

    definitions = configuration_session.exec(
        select(SettingDefinition)
    ).all()

    categories = {
        category.id: category.category_key
        for category in configuration_session.exec(
            select(SettingCategory)
        ).all()
    }

    definitions_by_key = {
        (
            categories[definition.category_id],
            definition.setting_key,
        ): definition
        for definition in definitions
    }

    options = {
        option.id: option.option_key
        for option in configuration_session.exec(
            select(SettingOption)
        ).all()
    }

    values = configuration_session.exec(
        select(ConfigurationValue).where(
            ConfigurationValue.revision_id == result.revision_id
        )
    ).all()

    values_by_definition = {
        value.setting_definition_id: value
        for value in values
    }

    provider_definition = definitions_by_key[
        ("rag.generation", "provider")
    ]
    model_definition = definitions_by_key[
        ("rag.generation", "model")
    ]
    temperature_definition = definitions_by_key[
        ("rag.generation", "temperature")
    ]
    top_k_definition = definitions_by_key[
        ("rag.retrieval", "top_k")
    ]
    builder_definition = definitions_by_key[
        ("rag.context", "builder")
    ]

    provider_value = values_by_definition[provider_definition.id]
    model_value = values_by_definition[model_definition.id]
    temperature_value = values_by_definition[
        temperature_definition.id
    ]
    top_k_value = values_by_definition[top_k_definition.id]
    builder_value = values_by_definition[builder_definition.id]

    assert provider_value.selected_option_id is not None
    assert options[provider_value.selected_option_id] == "ollama"

    assert model_value.selected_option_id is not None
    assert options[model_value.selected_option_id] == "mistral"

    assert temperature_value.value_float == 0.0
    assert top_k_value.value_integer == 5

    assert builder_value.selected_option_id is not None
    assert options[builder_value.selected_option_id] == "ranked"


def test_bootstrap_is_idempotent(
    configuration_session: Session,
) -> None:
    first = bootstrap_rag_configuration(
        configuration_session,
        actor_display_name="Premier bootstrap",
    )

    second = bootstrap_rag_configuration(
        configuration_session,
        actor_display_name="Deuxième bootstrap",
    )

    assert first.created is True
    assert second.created is False

    assert second.profile_id == first.profile_id
    assert second.revision_id == first.revision_id

    profiles = configuration_session.exec(
        select(ConfigurationProfile)
    ).all()

    revisions = configuration_session.exec(
        select(ConfigurationProfileRevision)
    ).all()

    assert len(profiles) == 1
    assert len(revisions) == 1
