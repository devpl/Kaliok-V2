from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "secret_value",
        "client_secret",
        "private_key",
        "authorization",
    }
)


class SensitiveConfigurationError(ValueError):
    """Raised when a new runtime configuration contains a secret-like key."""


def validate_endpoint(endpoint: str | None) -> str | None:
    """Reject URL userinfo so credentials cannot be embedded in an endpoint."""
    if endpoint is None:
        return None
    if not isinstance(endpoint, str):
        raise TypeError("Un endpoint doit être une chaîne ou null.")
    parsed = urlsplit(endpoint)
    if parsed.username is not None or parsed.password is not None:
        raise SensitiveConfigurationError(
            "Endpoint refusé: il ne doit pas contenir de credential intégré."
        )
    return endpoint


def _is_sensitive_key(key: object) -> bool:
    return isinstance(key, str) and key.strip().lower() in SENSITIVE_KEYS


def sanitize_audit_state(value: Any) -> Any:
    """Return a JSON-safe audit value with secret-like keys redacted recursively."""
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _is_sensitive_key(key) else sanitize_audit_state(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_audit_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Valeur d'audit non sérialisable: {type(value).__name__}.")


def _find_sensitive_key(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = f"{path}.{key}" if path else str(key)
            if _is_sensitive_key(key):
                return current
            found = _find_sensitive_key(item, current)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_sensitive_key(item, f"{path}[{index}]")
            if found:
                return found
    return None


def validate_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject secret-like keys in newly written resource/connection config."""
    if not isinstance(value, Mapping):
        raise TypeError("Une configuration doit être un objet JSON.")
    path = _find_sensitive_key(value)
    if path:
        raise SensitiveConfigurationError(
            f"Configuration refusée: clé sensible détectée à {path}. "
            "Utilisez une CredentialReference."
        )
    # Validate the full structure and return a detached plain dictionary.
    sanitized = sanitize_audit_state(value)
    return dict(sanitized)


def diff_states(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build a compact top-level/nested before-after diff for audit storage."""
    if before is None or after is None:
        return None
    changes: dict[str, Any] = {}
    keys = set(before) | set(after)
    for key in sorted(keys, key=str):
        old = before.get(key)
        new = after.get(key)
        if old == new:
            continue
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            nested = diff_states(old, new)
            if nested:
                changes[str(key)] = nested
        else:
            changes[str(key)] = {"before": old, "after": new}
    return changes or None
