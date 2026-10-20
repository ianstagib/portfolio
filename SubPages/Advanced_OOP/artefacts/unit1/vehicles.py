"""A small vehicle management system: Unit 1 of Advanced Object-Oriented
Design and Programming.

The module recaps the four pillars of object-oriented programming:

* Abstraction: ``Vehicle`` is an abstract base class. It says what every
  vehicle in the fleet can do without saying how each kind does it.
* Inheritance: ``Car``, ``Bike`` and ``Truck`` extend ``Vehicle`` and reuse
  its constructor, validation, odometer and ``describe()`` method.
* Polymorphism: each subclass answers ``wheels``, ``licence_category`` and
  ``_extra_details()`` in its own way, so ``Fleet`` can treat them alike.
* Encapsulation: state changes go through methods that protect each
  object's invariants. An odometer can only go forwards and a truck cannot
  be overloaded.

Access levels follow Python's conventions, because the language does not
enforce them: plain names are public, a single leading underscore marks a
protected name meant for the class and its subclasses, and a double leading
underscore is name-mangled to discourage access from outside the class.

Licence categories are simplified from the DVLA rules
(https://www.gov.uk/driving-licence-categories). Power-to-weight limits,
the higher category B weight allowed for zero-emission vans, trailers and
the date a licence was issued are all ignored.

>>> car = Car("ab12 cde", "Ford", "Focus", 2021, vin="WF0XXXGCDX1234567")
>>> car.registration
'AB12CDE'
>>> car.record_trip(42.5)
>>> car.odometer_km
42.5
>>> car.licence_category
'B'
>>> car
Car(registration='AB12CDE', make='Ford', model='Focus', year=2021)
"""

from __future__ import annotations

import abc
import datetime
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

__all__ = ["Vehicle", "Car", "Bike", "Truck", "Driver", "Fleet"]

# UK registrations are up to seven letters and digits once spaces are removed.
_REGISTRATION = re.compile(r"[A-Z0-9]{2,7}")
# A VIN is 17 characters and never uses I, O or Q (ISO 3779).
_VIN = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
# Higher licence categories include the lower ones in the same group.
_CATEGORIES_COVERED = {
    "A": {"A", "A2", "A1"},
    "A2": {"A2", "A1"},
    "A1": {"A1"},
    "B": {"B"},
    "C": {"C", "C1"},
    "C1": {"C1"},
}


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_positive(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive, finite number")
    return float(value)


class Vehicle(abc.ABC):
    """Abstract base class for every vehicle the fleet manages.

    Subclasses must say how many wheels they have and which licence
    category is needed to drive them. Everything else is shared.
    """

    # Registrations of vehicles whose finaliser has run. Kept so the
    # behaviour of __del__ can be observed and tested.
    decommissioned: ClassVar[list[str]] = []

    def __init__(
        self, registration: str, make: str, model: str, year: int, *, vin: str
    ) -> None:
        # Public: harmless descriptive data that callers may read and change.
        self.make = _require_text(make, "make")
        self.model = _require_text(model, "model")
        # Protected: read through properties, changed only by this class
        # and its subclasses.
        self._year = self._check_year(year)
        self._odometer_km = 0.0
        # Private: name-mangled to _Vehicle__vin. Only a masked copy is exposed.
        self.__vin = self._check_vin(vin)
        # Set last, so a vehicle that failed validation is never recorded as
        # decommissioned by __del__.
        self._registration = self._check_registration(registration)

    # Destructor (finaliser) ------------------------------------------------

    def __del__(self) -> None:
        # Python calls this when the last reference to the object goes, not
        # when `del` is used, and it is not guaranteed to run at interpreter
        # exit. It must not be relied on to release anything important.
        registration = getattr(self, "_registration", None)
        if registration is not None:
            type(self).decommissioned.append(registration)

    # Validation (protected helpers) --------------------------------------

    @staticmethod
    def _check_registration(registration: str) -> str:
        cleaned = _require_text(registration, "registration").replace(" ", "").upper()
        if not _REGISTRATION.fullmatch(cleaned):
            raise ValueError(f"invalid registration: {registration!r}")
        return cleaned

    @staticmethod
    def _check_vin(vin: str) -> str:
        cleaned = _require_text(vin, "vin").upper()
        if not _VIN.fullmatch(cleaned):
            # The value is deliberately left out of the message so that it
            # does not end up in logs.
            raise ValueError("invalid VIN: expected 17 characters, no I, O or Q")
        return cleaned

    @staticmethod
    def _check_year(year: int) -> int:
        if isinstance(year, bool) or not isinstance(year, int):
            raise TypeError("year must be an integer")
        latest = datetime.date.today().year + 1
        if not 1886 <= year <= latest:
            raise ValueError(f"year must be between 1886 and {latest}")
        return year

    # Read-only public interface ----------------------------------------

    @property
    def registration(self) -> str:
        return self._registration

    @property
    def year(self) -> int:
        return self._year

    @property
    def odometer_km(self) -> float:
        """Distance recorded so far. There is no setter, so it cannot be wound back."""
        return self._odometer_km

    @property
    def vin_masked(self) -> str:
        """The VIN with all but the last four characters hidden."""
        return "*" * 13 + self.__vin[-4:]

    def matches_vin(self, vin: str) -> bool:
        """Check a VIN supplied by someone else without revealing the stored one."""
        return isinstance(vin, str) and vin.strip().upper() == self.__vin

    # Behaviour ----------------------------------------------------------

    def record_trip(self, distance_km: float) -> None:
        """Add a completed trip to the odometer."""
        self._odometer_km += _require_positive(distance_km, "distance_km")

    def describe(self) -> str:
        """One line about the vehicle. Subclasses add detail via _extra_details()."""
        return (
            f"{self._year} {self.make} {self.model} ({self._registration}): "
            f"{self.wheels} wheels, licence {self.licence_category}"
            f"{self._extra_details()}, {self._odometer_km:,.1f} km"
        )

    def _extra_details(self) -> str:
        # Protected hook for subclasses. The base class adds nothing.
        return ""

    def __repr__(self) -> str:
        # The VIN is left out so that it does not leak into logs or tracebacks.
        return (
            f"{type(self).__name__}(registration={self._registration!r}, "
            f"make={self.make!r}, model={self.model!r}, year={self._year})"
        )

    # Abstract interface -------------------------------------------------

    @property
    @abc.abstractmethod
    def wheels(self) -> int:
        """Number of wheels."""

    @property
    @abc.abstractmethod
    def licence_category(self) -> str:
        """DVLA category needed to drive the vehicle."""


class Car(Vehicle):
    """A car with up to eight passenger seats, so drivable on a category B licence."""

    MAX_PASSENGER_SEATS: ClassVar[int] = 8

    def __init__(
        self,
        registration: str,
        make: str,
        model: str,
        year: int,
        *,
        vin: str,
        passenger_seats: int = 4,
    ) -> None:
        # The subclass checks its own arguments before the base constructor runs.
        if isinstance(passenger_seats, bool) or not isinstance(passenger_seats, int):
            raise TypeError("passenger_seats must be an integer")
        if not 1 <= passenger_seats <= self.MAX_PASSENGER_SEATS:
            raise ValueError(
                f"a car has 1 to {self.MAX_PASSENGER_SEATS} passenger seats; "
                "more than that is a minibus and needs a D1 or D licence"
            )
        super().__init__(registration, make, model, year, vin=vin)
        self._passenger_seats = passenger_seats

    @property
    def passenger_seats(self) -> int:
        return self._passenger_seats

    @property
    def wheels(self) -> int:
        return 4

    @property
    def licence_category(self) -> str:
        return "B"

    def _extra_details(self) -> str:
        return f", {self._passenger_seats} passenger seats"


class Bike(Vehicle):
    """A motorbike. The licence category depends on engine size and power."""

    def __init__(
        self,
        registration: str,
        make: str,
        model: str,
        year: int,
        *,
        vin: str,
        engine_cc: int,
        power_kw: float,
    ) -> None:
        if isinstance(engine_cc, bool) or not isinstance(engine_cc, int) or engine_cc < 0:
            raise ValueError("engine_cc must be a whole number of 0 or more (0 for electric)")
        power = _require_positive(power_kw, "power_kw")
        super().__init__(registration, make, model, year, vin=vin)
        self._engine_cc = engine_cc
        self._power_kw = power

    @property
    def engine_cc(self) -> int:
        return self._engine_cc

    @property
    def power_kw(self) -> float:
        return self._power_kw

    @property
    def wheels(self) -> int:
        return 2

    @property
    def licence_category(self) -> str:
        if self._engine_cc <= 125 and self._power_kw <= 11:
            return "A1"
        if self._power_kw <= 35:
            return "A2"
        return "A"

    def _extra_details(self) -> str:
        return f", {self._engine_cc} cc, {self._power_kw:g} kW"


class Truck(Vehicle):
    """A goods vehicle over 3,500 kg that carries a load up to its payload limit."""

    def __init__(
        self,
        registration: str,
        make: str,
        model: str,
        year: int,
        *,
        vin: str,
        gross_weight_kg: int,
        kerb_weight_kg: int,
        wheels: int = 6,
    ) -> None:
        if not isinstance(gross_weight_kg, int) or gross_weight_kg <= 3500:
            raise ValueError(
                "a truck must weigh more than 3,500 kg fully loaded; "
                "lighter goods vehicles are driven on a category B licence"
            )
        if not isinstance(kerb_weight_kg, int) or not 0 < kerb_weight_kg < gross_weight_kg:
            raise ValueError("kerb_weight_kg must be positive and below gross_weight_kg")
        if not isinstance(wheels, int) or wheels < 4 or wheels % 2:
            raise ValueError("a truck has an even number of wheels, at least 4")
        super().__init__(registration, make, model, year, vin=vin)
        self._gross_weight_kg = gross_weight_kg
        self._kerb_weight_kg = kerb_weight_kg
        self._wheels = wheels
        self._load_kg = 0.0

    @property
    def wheels(self) -> int:
        return self._wheels

    @property
    def licence_category(self) -> str:
        return "C1" if self._gross_weight_kg <= 7500 else "C"

    @property
    def payload_capacity_kg(self) -> int:
        return self._gross_weight_kg - self._kerb_weight_kg

    @property
    def load_kg(self) -> float:
        return self._load_kg

    def load(self, weight_kg: float) -> None:
        """Add cargo. Refuses, and changes nothing, if the truck would be overloaded."""
        weight = _require_positive(weight_kg, "weight_kg")
        if self._load_kg + weight > self.payload_capacity_kg:
            raise ValueError(
                f"overload: {self._load_kg + weight:,.0f} kg exceeds the "
                f"{self.payload_capacity_kg:,} kg payload"
            )
        self._load_kg += weight

    def unload(self, weight_kg: float) -> None:
        """Remove cargo. Cannot remove more than is on board."""
        weight = _require_positive(weight_kg, "weight_kg")
        if weight > self._load_kg:
            raise ValueError(f"only {self._load_kg:,.0f} kg is on board")
        self._load_kg -= weight

    def _extra_details(self) -> str:
        return f", {self._load_kg:,.0f} of {self.payload_capacity_kg:,} kg loaded"


@dataclass(frozen=True)
class Driver:
    """A driver and the licence categories they hold. Immutable once created."""

    name: str
    licence_categories: frozenset[str]

    def can_drive(self, vehicle: Vehicle) -> bool:
        # Fail-safe default: an unknown category entitles the driver to nothing.
        covered: set[str] = set()
        for category in self.licence_categories:
            covered |= _CATEGORIES_COVERED.get(category, set())
        return vehicle.licence_category in covered


class Fleet:
    """A collection of vehicles and their assigned drivers.

    Fleet *has* vehicles rather than *being* a list of them, so callers can
    only use the operations defined here.
    """

    def __init__(self, name: str) -> None:
        self.name = _require_text(name, "name")
        self._vehicles: dict[str, Vehicle] = {}
        self._drivers: dict[str, Driver] = {}

    def add(self, vehicle: Vehicle) -> None:
        if not isinstance(vehicle, Vehicle):
            raise TypeError("only Vehicle objects can join the fleet")
        if vehicle.registration in self._vehicles:
            raise ValueError(f"{vehicle.registration} is already in the fleet")
        self._vehicles[vehicle.registration] = vehicle

    def remove(self, registration: str) -> Vehicle:
        """Take a vehicle out of the fleet and return it."""
        key = Vehicle._check_registration(registration)
        self._drivers.pop(key, None)
        return self._vehicles.pop(key)

    def assign(self, registration: str, driver: Driver) -> None:
        """Assign a driver, but only if their licence covers the vehicle."""
        vehicle = self[registration]
        if not driver.can_drive(vehicle):
            raise PermissionError(
                f"{driver.name} is not licensed for category "
                f"{vehicle.licence_category} ({vehicle.registration})"
            )
        self._drivers[vehicle.registration] = driver

    def driver_of(self, registration: str) -> Driver | None:
        return self._drivers.get(Vehicle._check_registration(registration))

    def report(self) -> list[str]:
        # Polymorphism: one call, a different answer from each subclass.
        return [vehicle.describe() for vehicle in self]

    def total_distance_km(self) -> float:
        return sum(vehicle.odometer_km for vehicle in self)

    def __getitem__(self, registration: str) -> Vehicle:
        return self._vehicles[Vehicle._check_registration(registration)]

    def __contains__(self, registration: object) -> bool:
        return (
            isinstance(registration, str)
            and registration.replace(" ", "").upper() in self._vehicles
        )

    def __iter__(self) -> Iterator[Vehicle]:
        return iter(list(self._vehicles.values()))

    def __len__(self) -> int:
        return len(self._vehicles)
