from kaliok.audit.sanitization import (
    REDACTED,
    SensitiveConfigurationError,
    diff_states,
    sanitize_audit_state,
    validate_endpoint,
    validate_configuration,
)

__all__ = [
    "AuditActor", "AuditObject", "AuditService", "record_audit_event",
    "REDACTED",
    "SensitiveConfigurationError",
    "diff_states",
    "record_audit_event",
    "sanitize_audit_state",
    "validate_configuration",
    "validate_endpoint",
]


def __getattr__(name: str):
    if name in {"AuditActor", "AuditObject", "AuditService", "record_audit_event"}:
        from kaliok.audit.service import AuditActor, AuditObject, AuditService, record_audit_event

        return {
            "AuditActor": AuditActor,
            "AuditObject": AuditObject,
            "AuditService": AuditService,
            "record_audit_event": record_audit_event,
        }[name]
    raise AttributeError(name)
