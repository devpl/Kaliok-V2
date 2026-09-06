from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Session, select

from kaliok.storage.models import (
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ConfigurationValue,
    SettingCategory,
    SettingDefinition,
    SettingOption,
)


@dataclass(frozen=True)
class ConfigurationRevisionInfo:
    profile_id: UUID
    profile_key: str
    revision_id: UUID
    revision_number: int


class ConfigurationReader:
    def __init__(
        self,
        session: Session,
        *,
        profile_key: str = "production-default",
        revision_id: UUID | None = None,
    ) -> None:
        self._session = session
        if revision_id is None:
            self._profile = self._load_profile(profile_key)
            self._revision = self._load_active_revision(self._profile.id)
        else:
            self._revision = self._load_revision(revision_id)
            self._profile = self._load_profile_by_id(self._revision.profile_id)

    @property
    def revision_info(self) -> ConfigurationRevisionInfo:
        return ConfigurationRevisionInfo(
            profile_id=self._profile.id,
            profile_key=self._profile.profile_key,
            revision_id=self._revision.id,
            revision_number=self._revision.revision_number,
        )

    def get_string(
        self,
        category_key: str,
        setting_key: str,
    ) -> str:
        definition, value = self._get_definition_and_value(
            category_key,
            setting_key,
        )

        self._require_type(
            definition,
            expected_types={"string"},
        )

        if value.value_text is None:
            raise ValueError(
                self._missing_value_message(
                    category_key,
                    setting_key,
                )
            )

        return value.value_text

    def get_integer(
        self,
        category_key: str,
        setting_key: str,
    ) -> int:
        definition, value = self._get_definition_and_value(
            category_key,
            setting_key,
        )

        self._require_type(
            definition,
            expected_types={"integer"},
        )

        if value.value_integer is None:
            raise ValueError(
                self._missing_value_message(
                    category_key,
                    setting_key,
                )
            )

        return value.value_integer

    def get_float(
        self,
        category_key: str,
        setting_key: str,
    ) -> float:
        definition, value = self._get_definition_and_value(
            category_key,
            setting_key,
        )

        self._require_type(
            definition,
            expected_types={"float"},
        )

        if value.value_float is None:
            raise ValueError(
                self._missing_value_message(
                    category_key,
                    setting_key,
                )
            )

        return value.value_float

    def get_boolean(
        self,
        category_key: str,
        setting_key: str,
    ) -> bool:
        definition, value = self._get_definition_and_value(
            category_key,
            setting_key,
        )

        self._require_type(
            definition,
            expected_types={"boolean"},
        )

        if value.value_boolean is None:
            raise ValueError(
                self._missing_value_message(
                    category_key,
                    setting_key,
                )
            )

        return value.value_boolean

    def get_choice(
        self,
        category_key: str,
        setting_key: str,
    ) -> str:
        definition, value = self._get_definition_and_value(
            category_key,
            setting_key,
        )

        self._require_type(
            definition,
            expected_types={"choice"},
        )

        if value.selected_option_id is None:
            raise ValueError(
                self._missing_value_message(
                    category_key,
                    setting_key,
                )
            )

        option = self._session.get(
            SettingOption,
            value.selected_option_id,
        )

        if option is None:
            raise ValueError(
                "L'option sélectionnée n'existe plus pour "
                f"{category_key}.{setting_key}."
            )

        if option.setting_definition_id != definition.id:
            raise ValueError(
                "L'option sélectionnée n'appartient pas au paramètre "
                f"{category_key}.{setting_key}."
            )

        if not option.is_active:
            raise ValueError(
                "L'option sélectionnée est inactive pour "
                f"{category_key}.{setting_key}: {option.option_key}."
            )

        return option.option_key

    def _load_profile(
        self,
        profile_key: str,
    ) -> ConfigurationProfile:
        profile = self._session.exec(
            select(ConfigurationProfile).where(
                ConfigurationProfile.profile_key == profile_key
            )
        ).first()

        if profile is None:
            raise ValueError(
                f"Profil de configuration introuvable : {profile_key}."
            )

        if not profile.is_active:
            raise ValueError(
                f"Profil de configuration inactif : {profile_key}."
            )

        return profile

    def _load_active_revision(
        self,
        profile_id: UUID,
    ) -> ConfigurationProfileRevision:
        revisions = self._session.exec(
            select(ConfigurationProfileRevision)
            .where(
                ConfigurationProfileRevision.profile_id == profile_id,
                ConfigurationProfileRevision.status == "active",
            )
            .order_by(
                ConfigurationProfileRevision.revision_number.desc()
            )
        ).all()

        if not revisions:
            raise ValueError(
                "Le profil de configuration ne possède "
                "aucune révision active."
            )

        if len(revisions) > 1:
            raise ValueError(
                "Le profil de configuration possède plusieurs "
                "révisions actives."
            )

        return revisions[0]

    def _load_revision(
        self,
        revision_id: UUID,
    ) -> ConfigurationProfileRevision:
        revision = self._session.get(ConfigurationProfileRevision, revision_id)

        if revision is None:
            raise ValueError(
                f"Révision de configuration introuvable : {revision_id}."
            )

        return revision

    def _load_profile_by_id(self, profile_id: UUID) -> ConfigurationProfile:
        profile = self._session.get(ConfigurationProfile, profile_id)
        if profile is None:
            raise ValueError(f"Profil de configuration introuvable : {profile_id}.")
        if not profile.is_active:
            raise ValueError(
                f"Profil de configuration inactif : {profile.profile_key}."
            )
        return profile

    def _get_definition_and_value(
        self,
        category_key: str,
        setting_key: str,
    ) -> tuple[SettingDefinition, ConfigurationValue]:
        category = self._session.exec(
            select(SettingCategory).where(
                SettingCategory.category_key == category_key
            )
        ).first()

        if category is None:
            raise ValueError(
                f"Catégorie de configuration introuvable : {category_key}."
            )

        if not category.is_active:
            raise ValueError(
                f"Catégorie de configuration inactive : {category_key}."
            )

        definition = self._session.exec(
            select(SettingDefinition).where(
                SettingDefinition.category_id == category.id,
                SettingDefinition.setting_key == setting_key,
            )
        ).first()

        if definition is None:
            raise ValueError(
                "Paramètre de configuration introuvable : "
                f"{category_key}.{setting_key}."
            )

        value = self._session.exec(
            select(ConfigurationValue).where(
                ConfigurationValue.revision_id == self._revision.id,
                ConfigurationValue.setting_definition_id == definition.id,
            )
        ).first()

        if value is None:
            raise ValueError(
                self._missing_value_message(
                    category_key,
                    setting_key,
                )
            )

        return definition, value

    @staticmethod
    def _require_type(
        definition: SettingDefinition,
        *,
        expected_types: set[str],
    ) -> None:
        if definition.value_type not in expected_types:
            expected = ", ".join(sorted(expected_types))

            raise TypeError(
                f"Le paramètre {definition.setting_key} est de type "
                f"{definition.value_type}, attendu : {expected}."
            )

    @staticmethod
    def _missing_value_message(
        category_key: str,
        setting_key: str,
    ) -> str:
        return (
            "Valeur de configuration absente : "
            f"{category_key}.{setting_key}."
        )


class ConfigurationRevisionService:
    """Version and activate configuration profiles without mutating old revisions."""

    EDITABLE_KEYS = {"model", "temperature", "top_k", "provider", "builder"}

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_from(
        self,
        source_revision_id: UUID,
        values: dict[str, object],
        *,
        change_reason: str | None = None,
        actor_name: str = "Interface Kaliok",
    ) -> ConfigurationProfileRevision:
        source = self._session.get(ConfigurationProfileRevision, source_revision_id)
        if source is None:
            raise ValueError("Version source introuvable.")
        unknown = set(values) - self.EDITABLE_KEYS
        if unknown:
            raise ValueError(f"Paramètre non modifiable : {', '.join(sorted(unknown))}.")

        revisions = self._session.exec(
            select(ConfigurationProfileRevision).where(
                ConfigurationProfileRevision.profile_id == source.profile_id
            )
        ).all()
        revision = ConfigurationProfileRevision(
            profile_id=source.profile_id,
            revision_number=max((item.revision_number for item in revisions), default=0) + 1,
            status="draft",
            change_reason=change_reason,
            created_by_display_name=actor_name,
        )
        self._session.add(revision)
        self._session.flush()

        source_values = self._session.exec(
            select(ConfigurationValue).where(ConfigurationValue.revision_id == source.id)
        ).all()
        definitions = {
            item.id: item for item in self._session.exec(select(SettingDefinition)).all()
        }
        options = {
            item.id: item for item in self._session.exec(select(SettingOption)).all()
        }
        for old in source_values:
            definition = definitions[old.setting_definition_id]
            new = ConfigurationValue(
                revision_id=revision.id,
                setting_definition_id=old.setting_definition_id,
                value_text=old.value_text,
                value_integer=old.value_integer,
                value_float=old.value_float,
                value_boolean=old.value_boolean,
                selected_option_id=old.selected_option_id,
            )
            if definition.setting_key in values:
                value = values[definition.setting_key]
                if definition.value_type == "integer":
                    new.value_integer = int(value)
                elif definition.value_type == "float":
                    new.value_float = float(value)
                elif definition.value_type == "choice":
                    match = next(
                        (
                            option for option in options.values()
                            if option.setting_definition_id == definition.id
                            and option.option_key == str(value)
                            and option.is_active
                        ),
                        None,
                    )
                    if match is None:
                        raise ValueError(f"Option indisponible pour {definition.setting_key}.")
                    new.selected_option_id = match.id
                else:
                    new.value_text = str(value)
            self._session.add(new)
        self._session.commit()
        self._session.refresh(revision)
        return revision

    def activate(
        self, revision_id: UUID, *, actor_name: str = "Interface Kaliok"
    ) -> ConfigurationProfileRevision:
        revision = self._session.get(ConfigurationProfileRevision, revision_id)
        if revision is None:
            raise ValueError("Version introuvable.")
        active = self._session.exec(
            select(ConfigurationProfileRevision).where(
                ConfigurationProfileRevision.profile_id == revision.profile_id,
                ConfigurationProfileRevision.status == "active",
            )
        ).all()
        now = datetime.now(timezone.utc)
        for item in active:
            if item.id != revision.id:
                item.status = "retired"
                self._session.add(item)
        # The partial unique index is immediate: retire first, while keeping
        # both state changes inside the same transaction.
        self._session.flush()
        revision.status = "active"
        revision.activated_at = now
        revision.activated_by_display_name = actor_name
        self._session.add(revision)
        self._session.commit()
        self._session.refresh(revision)
        return revision
