from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from kaliok.storage.models import (
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ConfigurationValue,
    SettingCategory,
    SettingDefinition,
    SettingOption,
    utc_now,
)


@dataclass(frozen=True)
class BootstrapResult:
    profile_id: UUID
    revision_id: UUID
    created: bool


def _get_or_create_category(
    session: Session,
    *,
    category_key: str,
    label: str,
    description: str,
    display_order: int,
    parent_category_id: UUID | None = None,
) -> SettingCategory:
    category = session.exec(
        select(SettingCategory).where(
            SettingCategory.category_key == category_key
        )
    ).first()

    if category is not None:
        return category

    category = SettingCategory(
        category_key=category_key,
        label=label,
        description=description,
        display_order=display_order,
        parent_category_id=parent_category_id,
    )
    session.add(category)
    session.flush()

    return category


def _get_or_create_definition(
    session: Session,
    *,
    category_id: UUID,
    setting_key: str,
    label: str,
    description: str,
    value_type: str,
    display_order: int,
    is_required: bool = True,
    is_editable: bool = True,
    is_encrypted: bool = False,
) -> SettingDefinition:
    definition = session.exec(
        select(SettingDefinition).where(
            SettingDefinition.category_id == category_id,
            SettingDefinition.setting_key == setting_key,
        )
    ).first()

    if definition is not None:
        return definition

    definition = SettingDefinition(
        category_id=category_id,
        setting_key=setting_key,
        label=label,
        description=description,
        value_type=value_type,
        display_order=display_order,
        is_required=is_required,
        is_editable=is_editable,
        is_encrypted=is_encrypted,
    )
    session.add(definition)
    session.flush()

    return definition


def _get_or_create_option(
    session: Session,
    *,
    definition_id: UUID,
    option_key: str,
    label: str,
    description: str,
    display_order: int,
) -> SettingOption:
    option = session.exec(
        select(SettingOption).where(
            SettingOption.setting_definition_id == definition_id,
            SettingOption.option_key == option_key,
        )
    ).first()

    if option is not None:
        return option

    option = SettingOption(
        setting_definition_id=definition_id,
        option_key=option_key,
        label=label,
        description=description,
        display_order=display_order,
    )
    session.add(option)
    session.flush()

    return option


def bootstrap_rag_configuration(
    session: Session,
    *,
    actor_type: str = "system",
    actor_user_id: str | None = None,
    actor_display_name: str = "kaliok bootstrap",
) -> BootstrapResult:
    rag = _get_or_create_category(
        session,
        category_key="rag",
        label="RAG",
        description="Réglages du moteur Retrieval-Augmented Generation.",
        display_order=100,
    )

    generation = _get_or_create_category(
        session,
        category_key="rag.generation",
        label="Génération",
        description="Réglages de génération de la réponse.",
        display_order=100,
        parent_category_id=rag.id,
    )

    retrieval = _get_or_create_category(
        session,
        category_key="rag.retrieval",
        label="Recherche",
        description="Réglages de recherche des éléments de contexte.",
        display_order=200,
        parent_category_id=rag.id,
    )

    context = _get_or_create_category(
        session,
        category_key="rag.context",
        label="Contexte",
        description="Réglages de construction du contexte envoyé au modèle.",
        display_order=300,
        parent_category_id=rag.id,
    )

    provider_definition = _get_or_create_definition(
        session,
        category_id=generation.id,
        setting_key="provider",
        label="Fournisseur",
        description="Fournisseur utilisé pour la génération.",
        value_type="choice",
        display_order=100,
    )

    model_definition = _get_or_create_definition(
        session,
        category_id=generation.id,
        setting_key="model",
        label="Modèle de génération",
        description="Modèle utilisé pour générer la réponse.",
        value_type="choice",
        display_order=200,
    )

    temperature_definition = _get_or_create_definition(
        session,
        category_id=generation.id,
        setting_key="temperature",
        label="Température",
        description="Température utilisée pendant la génération.",
        value_type="float",
        display_order=300,
    )

    top_k_definition = _get_or_create_definition(
        session,
        category_id=retrieval.id,
        setting_key="top_k",
        label="Nombre de résultats",
        description="Nombre de candidats conservés par la recherche.",
        value_type="integer",
        display_order=100,
    )

    context_builder_definition = _get_or_create_definition(
        session,
        category_id=context.id,
        setting_key="builder",
        label="Construction du contexte",
        description="Méthode utilisée pour construire le contexte RAG.",
        value_type="choice",
        display_order=100,
    )

    ollama = _get_or_create_option(
        session,
        definition_id=provider_definition.id,
        option_key="ollama",
        label="Ollama",
        description="Génération locale via Ollama.",
        display_order=100,
    )

    mistral = _get_or_create_option(
        session,
        definition_id=model_definition.id,
        option_key="mistral",
        label="Mistral",
        description="Modèle Mistral disponible dans Ollama.",
        display_order=100,
    )

    _get_or_create_option(
        session,
        definition_id=model_definition.id,
        option_key="gemma3:4b",
        label="Gemma 3 4B",
        description="Modèle Gemma 3 4B disponible dans Ollama.",
        display_order=200,
    )

    _get_or_create_option(
        session,
        definition_id=model_definition.id,
        option_key="qwen3:8b",
        label="Qwen 3 8B",
        description="Modèle Qwen 3 8B disponible dans Ollama.",
        display_order=300,
    )

    _get_or_create_option(
        session,
        definition_id=model_definition.id,
        option_key="phi4-mini",
        label="Phi-4 Mini",
        description="Modèle Phi-4 Mini disponible dans Ollama.",
        display_order=400,
    )

    ranked = _get_or_create_option(
        session,
        definition_id=context_builder_definition.id,
        option_key="ranked",
        label="Contexte classé",
        description=(
            "Construit le contexte à partir des résultats classés "
            "par pertinence."
        ),
        display_order=100,
    )

    profile = session.exec(
        select(ConfigurationProfile).where(
            ConfigurationProfile.profile_key == "production-default"
        )
    ).first()

    if profile is not None:
        revision = session.exec(
            select(ConfigurationProfileRevision)
            .where(
                ConfigurationProfileRevision.profile_id == profile.id,
                ConfigurationProfileRevision.status == "active",
            )
            .order_by(
                ConfigurationProfileRevision.revision_number.desc()
            )
        ).first()

        if revision is None:
            raise ValueError(
                "Le profil production-default existe sans révision active."
            )

        return BootstrapResult(
            profile_id=profile.id,
            revision_id=revision.id,
            created=False,
        )

    profile = ConfigurationProfile(
        profile_key="production-default",
        label="Production par défaut",
        description="Configuration RAG active par défaut.",
        is_active=True,
        is_default=True,
    )
    session.add(profile)
    session.flush()

    activated_at = utc_now()

    revision = ConfigurationProfileRevision(
        profile_id=profile.id,
        revision_number=1,
        status="active",
        change_reason="Initialisation de la configuration RAG.",
        created_by_actor_type=actor_type,
        created_by_user_id=actor_user_id,
        created_by_display_name=actor_display_name,
        activated_by_user_id=actor_user_id,
        activated_by_display_name=actor_display_name,
        activated_at=activated_at,
    )
    session.add(revision)
    session.flush()

    session.add_all(
        [
            ConfigurationValue(
                revision_id=revision.id,
                setting_definition_id=provider_definition.id,
                selected_option_id=ollama.id,
            ),
            ConfigurationValue(
                revision_id=revision.id,
                setting_definition_id=model_definition.id,
                selected_option_id=mistral.id,
            ),
            ConfigurationValue(
                revision_id=revision.id,
                setting_definition_id=temperature_definition.id,
                value_float=0.0,
            ),
            ConfigurationValue(
                revision_id=revision.id,
                setting_definition_id=top_k_definition.id,
                value_integer=5,
            ),
            ConfigurationValue(
                revision_id=revision.id,
                setting_definition_id=context_builder_definition.id,
                selected_option_id=ranked.id,
            ),
        ]
    )

    session.flush()

    return BootstrapResult(
        profile_id=profile.id,
        revision_id=revision.id,
        created=True,
    )