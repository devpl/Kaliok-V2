from __future__ import annotations

from kaliok.observability.events import ObservabilityEvent


class NoOpObserver:
    def emit(self, event: ObservabilityEvent) -> None:
        return None
