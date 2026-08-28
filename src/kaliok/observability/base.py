from __future__ import annotations

from typing import Protocol

from kaliok.observability.events import ObservabilityEvent


class Observer(Protocol):
    def emit(self, event: ObservabilityEvent) -> None: ...
