from __future__ import annotations

from collections.abc import Iterable

from kaliok.observability.base import Observer
from kaliok.observability.events import ObservabilityEvent


class CompositeObserver:
    def __init__(self, observers: Iterable[Observer]) -> None:
        self._observers = tuple(observers)

    def emit(self, event: ObservabilityEvent) -> None:
        for observer in self._observers:
            try:
                observer.emit(event)
            except Exception:
                # L'observabilité ne doit jamais modifier le traitement observé.
                continue
