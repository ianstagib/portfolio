"""The Unit 3 car manufacturing system, built with the Factory Method pattern.

The pattern has four roles, and each has a class here:

* Product: ``Car``, an abstract class whose subclasses must implement
  ``drive()``.
* Concrete products: ``Sedan``, ``SUV`` and ``Hatchback``.
* Creator: ``CarFactory``, an abstract class with the factory method
  ``create_car()``. It also has ``produce()``, its own operation, which calls
  the factory method, checks what comes back and stamps a serial number on it.
  ``produce()`` never names a concrete car.
* Concrete creators: ``SedanFactory``, ``SUVFactory`` and
  ``HatchbackFactory``, each of which overrides ``create_car()``.

The client functions ``road_test()`` and ``run_production()`` depend only on
``Car`` and ``CarFactory``. The one place that turns a name into a concrete
factory is ``factory_for()``, which ``main()`` uses when it assembles the
program.

>>> serials = SerialNumbers()
>>> record = SUVFactory(serials).produce()
>>> record.serial, record.car.body_type, record.factory
('CAR-000001', 'SUV', 'SUVFactory')
>>> run_production([SedanFactory(serials), HatchbackFactory(serials)])
['CAR-000002 Sedan: a smooth, quiet ride on the motorway', 'CAR-000003 Hatchback: nipping through town and into a tight parking space']
"""

from __future__ import annotations

import abc
import itertools
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

__all__ = [
    "Car",
    "Sedan",
    "SUV",
    "Hatchback",
    "SerialNumbers",
    "BuildRecord",
    "CarFactory",
    "SedanFactory",
    "SUVFactory",
    "HatchbackFactory",
    "FACTORIES",
    "factory_for",
    "road_test",
    "run_production",
    "main",
]


# --- Product and concrete products -------------------------------------------


class Car(abc.ABC):
    """The product. Every car can be driven and can say what it is.

    ``drive()`` returns a sentence rather than printing it, so that callers
    and tests can use the result.
    """

    body_type: ClassVar[str]
    seats: ClassVar[int]

    @abc.abstractmethod
    def drive(self) -> str:
        """Return a one-line description of driving this car."""

    def describe(self) -> str:
        """Shared behaviour that subclasses inherit and do not override."""
        return f"{self.body_type} with {self.seats} seats"


class Sedan(Car):
    body_type = "Sedan"
    seats = 5

    def drive(self) -> str:
        return "Sedan: a smooth, quiet ride on the motorway"


class SUV(Car):
    body_type = "SUV"
    seats = 7

    def drive(self) -> str:
        return "SUV: high driving position and four-wheel drive on a gravel track"


class Hatchback(Car):
    body_type = "Hatchback"
    seats = 5

    def drive(self) -> str:
        return "Hatchback: nipping through town and into a tight parking space"


# --- Supporting types ---------------------------------------------------------


class SerialNumbers:
    """A source of unique serial numbers that several factories can share.

    It is passed in to each factory rather than held in a class attribute, so
    a test or a second production line can have its own sequence. The lock is
    taken only inside the public method, following the Thread-Safe Interface
    pattern, so the numbers stay unique when factories run in parallel threads.

    >>> serials = SerialNumbers(prefix="TEST", start=41)
    >>> serials.next(), serials.next()
    ('TEST-000041', 'TEST-000042')
    """

    def __init__(self, prefix: str = "CAR", start: int = 1) -> None:
        self._prefix = prefix
        self._counter = itertools.count(start)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            number = next(self._counter)
        return f"{self._prefix}-{number:06d}"


@dataclass(frozen=True)
class BuildRecord:
    """What ``CarFactory.produce()`` returns: the car and its paperwork."""

    serial: str
    car: Car
    factory: str


# --- Creator and concrete creators --------------------------------------------


class CarFactory(abc.ABC):
    """The creator. Subclasses decide which car ``create_car()`` returns.

    ``produce()`` is the creator's own operation. It is written once, here,
    against the abstract ``Car``, and works unchanged for every subclass.
    """

    def __init__(self, serials: SerialNumbers | None = None) -> None:
        self._serials = serials if serials is not None else SerialNumbers()

    @abc.abstractmethod
    def create_car(self) -> Car:
        """The factory method. Return a new, unregistered car."""

    def produce(self) -> BuildRecord:
        """Make a car with the factory method, check it and give it a serial."""
        car = self.create_car()
        if not isinstance(car, Car):
            raise TypeError(
                f"{type(self).__name__}.create_car() returned "
                f"{type(car).__name__}, not a Car"
            )
        return BuildRecord(self._serials.next(), car, type(self).__name__)


class SedanFactory(CarFactory):
    def create_car(self) -> Car:
        return Sedan()


class SUVFactory(CarFactory):
    def create_car(self) -> Car:
        return SUV()


class HatchbackFactory(CarFactory):
    def create_car(self) -> Car:
        return Hatchback()


# --- Client code: depends only on Car and CarFactory ---------------------------


def road_test(factory: CarFactory) -> str:
    """Build one car with whatever factory is given and take it for a drive."""
    record = factory.produce()
    return f"{record.serial} {record.car.drive()}"


def run_production(factories: Iterable[CarFactory]) -> list[str]:
    """Road test one car from each factory, in order."""
    return [road_test(factory) for factory in factories]


# --- Composition root: the one place that maps names to concrete factories ----

FactoryMaker = Callable[[SerialNumbers], CarFactory]

#: Read-only, so no other module can add or replace a factory at run time.
FACTORIES: Mapping[str, FactoryMaker] = MappingProxyType(
    {
        "sedan": SedanFactory,
        "suv": SUVFactory,
        "hatchback": HatchbackFactory,
    }
)


def factory_for(
    name: str,
    serials: SerialNumbers,
    registry: Mapping[str, FactoryMaker] = FACTORIES,
) -> CarFactory:
    """Turn a name from an order or a configuration file into a factory.

    A different ``registry`` can be passed in to add car types without
    editing this module.

    >>> factory_for("Hatchback", SerialNumbers()).produce().car.body_type
    'Hatchback'
    >>> factory_for("van", SerialNumbers())
    Traceback (most recent call last):
    ...
    ValueError: unknown car type 'van'; choose from hatchback, sedan, suv
    """
    try:
        maker = registry[name.strip().lower()]
    except KeyError:
        options = ", ".join(sorted(registry))
        raise ValueError(f"unknown car type {name!r}; choose from {options}") from None
    return maker(serials)


def main(order: Iterable[str] = ("sedan", "suv", "hatchback")) -> None:
    """Demonstration: build and drive one car of each type in ``order``."""
    serials = SerialNumbers()
    factories = [factory_for(name, serials) for name in order]
    for line in run_production(factories):
        print(line)


if __name__ == "__main__":
    main()
