"""Task 2: the Bridge pattern for devices and their remote controls.

Two hierarchies vary independently. ``Device`` is the implementor, with
``TV`` and ``Radio`` as concrete implementors. ``RemoteControl`` is the
abstraction, with ``AdvancedRemote`` as a refined abstraction. Each remote
holds a device and works only through the ``Device`` interface, so any remote
works with any device: 2 remotes x 2 devices need 4 classes, not 4 subclasses
such as ``AdvancedTVRemote``, and a third of each needs 6 classes, not 9.

>>> remote = AdvancedRemote(Radio())
>>> remote.toggle_power(); remote.set_channel(5); remote.mute()
>>> remote.device.status()
'Radio: on, station 5, volume 0'
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["Device", "TV", "Radio", "RemoteControl", "AdvancedRemote"]


class Device(ABC):
    """The implementor: what every device can do."""

    name = "Device"
    channel_label = "channel"
    max_channel = 99

    def __init__(self) -> None:
        self._on = False
        self._volume = 30
        self._channel = 1

    def is_enabled(self) -> bool:
        return self._on

    def enable(self) -> None:
        self._on = True

    def disable(self) -> None:
        self._on = False

    def get_volume(self) -> int:
        return self._volume

    def set_volume(self, percent: int) -> None:
        self._volume = max(0, min(100, percent))

    def get_channel(self) -> int:
        return self._channel

    def set_channel(self, channel: int) -> None:
        if not 1 <= channel <= self.max_channel:
            raise ValueError(f"{self.name} has {self.channel_label}s 1 to {self.max_channel}")
        self._channel = channel

    @abstractmethod
    def status(self) -> str: ...


class TV(Device):
    name, max_channel = "TV", 999

    def status(self) -> str:
        state = "on" if self._on else "off"
        return f"TV: {state}, channel {self._channel}, volume {self._volume}"


class Radio(Device):
    name, channel_label, max_channel = "Radio", "station", 20

    def status(self) -> str:
        state = "on" if self._on else "off"
        return f"Radio: {state}, station {self._channel}, volume {self._volume}"


class RemoteControl:
    """The abstraction: basic controls, written only against Device."""

    STEP = 10

    def __init__(self, device: Device) -> None:
        self.device = device

    def toggle_power(self) -> None:
        if self.device.is_enabled():
            self.device.disable()
        else:
            self.device.enable()

    def volume_up(self) -> None:
        self.device.set_volume(self.device.get_volume() + self.STEP)

    def volume_down(self) -> None:
        self.device.set_volume(self.device.get_volume() - self.STEP)

    def channel_up(self) -> None:
        next_channel = self.device.get_channel() % self.device.max_channel + 1
        self.device.set_channel(next_channel)


class AdvancedRemote(RemoteControl):
    """A refined abstraction: more controls, still only through Device."""

    def mute(self) -> None:
        self.device.set_volume(0)

    def set_channel(self, channel: int) -> None:
        self.device.set_channel(channel)
