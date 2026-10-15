"""A thread-safe publish/subscribe bus."""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from mizuoto.core.events.models import EventBase

logger = logging.getLogger(__name__)

Handler = Callable[[EventBase], None]


class EventBus(ABC):
    @abstractmethod
    def publish(self, event: EventBase) -> None:
        """Deliver ``event`` to every subscriber. Safe to call from any thread."""

    @abstractmethod
    def subscribe(self, handler: Handler) -> Callable[[], None]:
        """Register ``handler``; return a function that unsubscribes it."""


class LocalEventBus(EventBus):
    """Calls handlers synchronously on the publishing thread.

    A subscriber that must run elsewhere, such as on an asyncio loop, hands the event over itself.
    """

    def __init__(self) -> None:
        self._handlers: list[Handler] = []
        self._lock = threading.Lock()

    def publish(self, event: EventBase) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler failed for %s", event.__event_name__)

    def subscribe(self, handler: Handler) -> Callable[[], None]:
        with self._lock:
            self._handlers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe
