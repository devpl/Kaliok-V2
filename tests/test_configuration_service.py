from __future__ import annotations

import pytest
from sqlmodel import Session, select

from kaliok.configuration import (
    ConfigurationReader,
    bootstrap_rag_configuration,
)
from kaliok.storage.models import (
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ConfigurationValue,
    SettingCategory,
    SettingDefinition,
    SettingOption,
)


def test_reader_reads_active_profile_values(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(
        configuration_session,
        actor_type="user",
        actor_user_id="user-1",
        actor_display_name="Alice Martin",
    )

    reader = ConfigurationReader(configuration_session)

    assert reader.revision_info.profile_id == result.profile_id
    assert reader.revision_info.revision_id == result.revision_id
    assert reader.revision_info.profile_key == "production-default"
    assert reader.revision_info.revision_number == 1

    assert (
        reader.get_choice(
            "rag.generation",
            "provider",
        )
        == "ollama"
    )

    assert (
        reader.get_choice(
            "rag.generation",
            "model",
        )
        == "mistral"
    )

    assert (
        reader.get_float(
            "rag.generation",
            "temperature",
        )
        == 0.0
    )

    assert (
        reader.get_integer(
            "rag.retrieval",
            "top_k",
        )
        == 5
    )

    assert (
        reader.get_choice(
            "rag.context",
            "builder",
        )
        == "ranked"
    )


@pytest.mark.parametrize("revision_status", ["draft", "retired"])
def test_reader_can_select_an_explicit_revision_without_profile_key(
    configuration_session: Session,
    revision_status: str,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)
    revision = configuration_session.get(
        ConfigurationProfileRevision,
        result.revision_id,
    )
    assert revision is not None
    revision.status = revision_status
    configuration_session.add(revision)
    configuration_session.flush()

    reader = ConfigurationReader(
        configuration_session,
        profile_key="ce-profil-ne-doit-pas-etre-consulte",
        revision_id=result.revision_id,
    )

    assert reader.revision_info.revision_id == result.revision_id
    assert reader.get_integer("rag.retrieval", "top_k") == 5


def test_reader_rejects_unknown_profile(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    with pytest.raises(
        ValueError,
        match="Profil de configuration introuvable",
    ):
        ConfigurationReader(
            configuration_session,
            profile_key="unknown-profile",
        )


def test_reader_rejects_inactive_profile(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

    profile = configuration_session.get(
        ConfigurationProfile,
        result.profile_id,
    )
    assert profile is not None

    profile.is_active = False
    configuration_session.add(profile)
    configuration_session.flush()

    with pytest.raises(
        ValueError,
        match="Profil de configuration inactif",
    ):
        ConfigurationReader(configuration_session)


def test_reader_rejects_missing_setting(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    reader = ConfigurationReader(configuration_session)

    with pytest.raises(
        ValueError,
        match="Paramètre de configuration introuvable",
    ):
        reader.get_integer(
            "rag.retrieval",
            "missing",
        )


def test_reader_rejects_missing_category(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    reader = ConfigurationReader(configuration_session)

    with pytest.raises(
        ValueError,
        match="Catégorie de configuration introuvable",
    ):
        reader.get_integer(
            "rag.unknown",
            "top_k",
        )


def test_reader_rejects_wrong_type(
    configuration_session: Session,
) -> None:
    bootstrap_rag_configuration(configuration_session)

    reader = ConfigurationReader(configuration_session)

    with pytest.raises(
        TypeError,
        match="est de type integer",
    ):
        reader.get_choice(
            "rag.retrieval",
            "top_k",
        )


def test_reader_rejects_missing_value(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

    retrieval_category = configuration_session.exec(
        select(SettingCategory).where(
            SettingCategory.category_key == "rag.retrieval"
        )
    ).one()

    top_k_definition = configuration_session.exec(
        select(SettingDefinition).where(
            SettingDefinition.category_id == retrieval_category.id,
            SettingDefinition.setting_key == "top_k",
        )
    ).one()

    value = configuration_session.exec(
        select(ConfigurationValue).where(
            ConfigurationValue.revision_id == result.revision_id,
            ConfigurationValue.setting_definition_id
            == top_k_definition.id,
        )
    ).one()

    configuration_session.delete(value)
    configuration_session.flush()

    reader = ConfigurationReader(configuration_session)

    with pytest.raises(
        ValueError,
        match="Valeur de configuration absente",
    ):
        reader.get_integer(
            "rag.retrieval",
            "top_k",
        )


def test_reader_rejects_inactive_choice(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

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

    model_value = configuration_session.exec(
        select(ConfigurationValue).where(
            ConfigurationValue.revision_id == result.revision_id,
            ConfigurationValue.setting_definition_id
            == model_definition.id,
        )
    ).one()

    assert model_value.selected_option_id is not None

    option = configuration_session.get(
        SettingOption,
        model_value.selected_option_id,
    )
    assert option is not None

    option.is_active = False
    configuration_session.add(option)
    configuration_session.flush()

    reader = ConfigurationReader(configuration_session)

    with pytest.raises(
        ValueError,
        match="L'option sélectionnée est inactive",
    ):
        reader.get_choice(
            "rag.generation",
            "model",
        )


def test_reader_rejects_choice_from_another_setting(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

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

    provider_definition = configuration_session.exec(
        select(SettingDefinition).where(
            SettingDefinition.category_id == generation_category.id,
            SettingDefinition.setting_key == "provider",
        )
    ).one()

    model_value = configuration_session.exec(
        select(ConfigurationValue).where(
            ConfigurationValue.revision_id == result.revision_id,
            ConfigurationValue.setting_definition_id
            == model_definition.id,
        )
    ).one()

    provider_option = configuration_session.exec(
        select(SettingOption).where(
            SettingOption.setting_definition_id
            == provider_definition.id,
            SettingOption.option_key == "ollama",
        )
    ).one()

    model_value.selected_option_id = provider_option.id
    configuration_session.add(model_value)
    configuration_session.flush()

    reader = ConfigurationReader(configuration_session)

    with pytest.raises(
        ValueError,
        match="n'appartient pas au paramètre",
    ):
        reader.get_choice(
            "rag.generation",
            "model",
        )


def test_database_rejects_multiple_active_revisions(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

    second_revision = ConfigurationProfileRevision(
        profile_id=result.profile_id,
        revision_number=2,
        status="active",
        change_reason="Test d'une configuration incohérente.",
        created_by_actor_type="user",
        created_by_user_id="user-2",
        created_by_display_name="Bob Martin",
        activated_by_user_id="user-2",
        activated_by_display_name="Bob Martin",
    )

    configuration_session.add(second_revision)

    with pytest.raises(Exception):
        configuration_session.flush()


def test_reader_rejects_profile_without_active_revision(
    configuration_session: Session,
) -> None:
    result = bootstrap_rag_configuration(configuration_session)

    revision = configuration_session.get(
        ConfigurationProfileRevision,
        result.revision_id,
    )
    assert revision is not None

    revision.status = "retired"
    configuration_session.add(revision)
    configuration_session.flush()

    with pytest.raises(
        ValueError,
        match="aucune révision active",
    ):
        ConfigurationReader(configuration_session)
