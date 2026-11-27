"""Case study support: the Observer pattern in a small security monitor.

A ``LoginMonitor`` is the subject. Each login attempt it records is an event,
and observers react to it: an audit log keeps every event, a lockout policy
locks an account after repeated failures, and an alert raises a message when
a locked account is tried again. The monitor knows only the ``Observer``
interface, so a new response is added by subscribing it, not by editing the
monitor.

Three design decisions from Nesteruk (2022, ch. 20) are made explicitly:

* Notification works on a copy of the subscriber list, so an observer may
  unsubscribe while it is being notified (the reentrancy problem).
* Subscribing the same observer twice has no effect.
* Observers marked as required cannot be unsubscribed, because switching off
  monitoring is exactly what an attacker wants. One failing observer is
  reported and does not stop the others being notified.

>>> monitor = LoginMonitor()
>>> audit, lockout = AuditLog(), LockoutPolicy(threshold=3)
>>> monitor.subscribe(audit, required=True)
>>> monitor.subscribe(lockout)
>>> for _ in range(3):
...     _ = monitor.record("alice", success=False, source="203.0.113.7")
>>> lockout.is_locked("alice"), len(audit.events)
(True, 3)
"""

from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = [
    "LoginEvent",
    "Observer",
    "LoginMonitor",
    "AuditLog",
    "LockoutPolicy",
    "LockedAccountAlert",
    "MonitoringError",
]


class MonitoringError(RuntimeError):
    """An attempt to weaken monitoring, such as removing a required observer."""


@dataclass(frozen=True)
class LoginEvent:
    """What the subject tells its observers. Frozen, so no observer can alter it."""

    user: str
    success: bool
    source: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Observer(abc.ABC):
    @abc.abstractmethod
    def notify(self, event: LoginEvent, monitor: LoginMonitor) -> None:
        """React to one login event."""


class LoginMonitor:
    """The subject: records login attempts and notifies every subscriber."""

    def __init__(self) -> None:
        self._observers: list[Observer] = []
        self._required: set[int] = set()
        self.errors: list[str] = []

    def subscribe(self, observer: Observer, required: bool = False) -> None:
        if not isinstance(observer, Observer):
            raise TypeError("observer must implement Observer")
        if observer not in self._observers:
            self._observers.append(observer)
        if required:
            self._required.add(id(observer))

    def unsubscribe(self, observer: Observer) -> None:
        if id(observer) in self._required:
            raise MonitoringError(f"{type(observer).__name__} is required and stays subscribed")
        if observer in self._observers:
            self._observers.remove(observer)

    @property
    def observers(self) -> tuple[Observer, ...]:
        return tuple(self._observers)

    def record(self, user: str, success: bool, source: str) -> LoginEvent:
        event = LoginEvent(user, success, source)
        for observer in list(self._observers):  # a copy: observers may unsubscribe
            try:
                observer.notify(event, self)
            except Exception as error:  # one broken observer must not silence the rest
                self.errors.append(f"{type(observer).__name__}: {error}")
        return event


class AuditLog(Observer):
    def __init__(self) -> None:
        self.events: list[LoginEvent] = []

    def notify(self, event: LoginEvent, monitor: LoginMonitor) -> None:
        self.events.append(event)


class LockoutPolicy(Observer):
    """Locks an account after ``threshold`` failures in a row."""

    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold
        self._failures: Counter[str] = Counter()
        self._locked: set[str] = set()

    def is_locked(self, user: str) -> bool:
        return user in self._locked

    def notify(self, event: LoginEvent, monitor: LoginMonitor) -> None:
        if event.success:
            self._failures[event.user] = 0
            return
        self._failures[event.user] += 1
        if self._failures[event.user] >= self.threshold:
            self._locked.add(event.user)


class LockedAccountAlert(Observer):
    """Raises an alert when a locked account is tried again."""

    def __init__(self, policy: LockoutPolicy) -> None:
        self.policy = policy
        self.alerts: list[str] = []

    def notify(self, event: LoginEvent, monitor: LoginMonitor) -> None:
        if self.policy.is_locked(event.user):
            self.alerts.append(f"locked account {event.user!r} tried from {event.source}")
