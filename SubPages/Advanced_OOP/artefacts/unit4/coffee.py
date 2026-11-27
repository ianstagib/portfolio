"""The Unit 4 coffee shop, built with the Decorator pattern.

The brief asks for a coffee shop where toppings such as milk and sugar are
added to a base coffee at run time. The pattern has four roles, and each has
a class here:

* Component: ``Beverage``, an abstract class with ``description()`` and
  ``cost()``.
* Concrete components: ``Coffee`` and ``Espresso``, the drinks that exist
  before any topping is added. Neither is edited to support toppings.
* Decorator: ``Topping``, an abstract ``Beverage`` that holds another
  ``Beverage`` and forwards every call to it.
* Concrete decorators: ``Milk``, ``OatMilk``, ``Sugar``, ``VanillaSyrup``,
  ``ExtraShot`` and ``WhippedCream``, which add their name and price, and
  ``LoyaltyDiscount``, which changes the price instead of adding to it.

Prices are whole pence, so totals never pick up floating-point error.
``make_drink()`` is the order desk: it turns names from an order into
decorators through one read-only menu, refuses anything not on it and
enforces the shop's limits, so callers never construct toppings from raw
input.

>>> drink = Sugar(Milk(Coffee()))
>>> drink.description(), format_price(drink.cost())
('Coffee (regular), milk, sugar', '£3.05')
>>> make_drink("espresso", ["extra shot", "oat milk"]).receipt_line()
'Espresso, extra shot, oat milk: £3.25'
"""

from __future__ import annotations

import abc
import functools
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import ClassVar, ParamSpec, TypeVar

__all__ = [
    "Beverage",
    "Coffee",
    "Espresso",
    "Topping",
    "Milk",
    "OatMilk",
    "Sugar",
    "VanillaSyrup",
    "ExtraShot",
    "WhippedCream",
    "LoyaltyDiscount",
    "BASES",
    "TOPPINGS",
    "MAX_TOPPINGS",
    "MAX_EACH",
    "OrderError",
    "format_price",
    "make_drink",
    "audited",
    "main",
]


def format_price(pence: int) -> str:
    """Show a price held in pence as pounds, for example 305 as '£3.05'."""
    return f"£{pence // 100}.{pence % 100:02d}"


# --- Component and concrete components ---------------------------------------


class Beverage(abc.ABC):
    """Anything the shop can sell as a drink."""

    __slots__ = ()

    @abc.abstractmethod
    def description(self) -> str:
        """What goes on the cup and the receipt."""

    @abc.abstractmethod
    def cost(self) -> int:
        """The price in pence."""

    def layers(self) -> tuple[str, ...]:
        """The base drink followed by each topping, innermost first."""
        return (self.description(),)

    def receipt_line(self) -> str:
        return f"{self.description()}: {format_price(self.cost())}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.receipt_line()}>"


class Coffee(Beverage):
    """The shop's existing filter coffee. It knows nothing about toppings."""

    PRICES: ClassVar[Mapping[str, int]] = MappingProxyType(
        {"small": 220, "regular": 250, "large": 280}
    )

    __slots__ = ("_size",)

    def __init__(self, size: str = "regular") -> None:
        if size not in self.PRICES:
            raise ValueError(f"unknown size {size!r}; choose from {sorted(self.PRICES)}")
        self._size = size

    @property
    def size(self) -> str:
        return self._size

    def description(self) -> str:
        return f"Coffee ({self._size})"

    def cost(self) -> int:
        return self.PRICES[self._size]


class Espresso(Beverage):
    """A second base drink, to show that toppings work with any Beverage."""

    __slots__ = ()

    def description(self) -> str:
        return "Espresso"

    def cost(self) -> int:
        return 210


# --- Decorator and concrete decorators ---------------------------------------


class Topping(Beverage):
    """Wraps a Beverage, forwards to it and adds one thing of its own.

    A Topping is itself a Beverage, so toppings can wrap one another to any
    depth and the client cannot tell a plain coffee from a decorated one.
    """

    name: ClassVar[str]
    price: ClassVar[int]

    __slots__ = ("_drink",)

    def __init__(self, drink: Beverage) -> None:
        if not hasattr(type(self), "name"):
            # No abstract method is left to stop this, so check by hand.
            raise TypeError(f"{type(self).__name__} is abstract; use a concrete topping")
        if not isinstance(drink, Beverage):
            raise TypeError(f"{type(self).__name__} can only wrap a Beverage, not {type(drink).__name__}")
        self._drink = drink

    @property
    def wrapped(self) -> Beverage:
        """The drink inside this layer."""
        return self._drink

    def description(self) -> str:
        return f"{self._drink.description()}, {self.name}"

    def cost(self) -> int:
        return self._drink.cost() + self.price

    def layers(self) -> tuple[str, ...]:
        return (*self._drink.layers(), self.name)


class Milk(Topping):
    name, price = "milk", 40
    __slots__ = ()


class OatMilk(Topping):
    name, price = "oat milk", 55
    __slots__ = ()


class Sugar(Topping):
    name, price = "sugar", 15
    __slots__ = ()


class VanillaSyrup(Topping):
    name, price = "vanilla syrup", 45
    __slots__ = ()


class ExtraShot(Topping):
    name, price = "extra shot", 60
    __slots__ = ()


class WhippedCream(Topping):
    name, price = "whipped cream", 50
    __slots__ = ()


class LoyaltyDiscount(Topping):
    """Takes 10% off everything inside it, rounded down to the penny.

    It alters the price rather than adding to it, so where it sits in the
    stack changes the total: toppings added outside it are not discounted.
    """

    name, price = "loyalty discount", 0
    __slots__ = ()

    def cost(self) -> int:
        inner = self._drink.cost()
        return inner - inner // 10


# --- The order desk ----------------------------------------------------------

BASES: Mapping[str, Callable[[], Beverage]] = MappingProxyType(
    {
        "coffee": Coffee,
        "small coffee": lambda: Coffee("small"),
        "large coffee": lambda: Coffee("large"),
        "espresso": Espresso,
    }
)

TOPPINGS: Mapping[str, type[Topping]] = MappingProxyType(
    {cls.name: cls for cls in (Milk, OatMilk, Sugar, VanillaSyrup, ExtraShot, WhippedCream)}
)

MAX_TOPPINGS = 5
MAX_EACH = 3


class OrderError(ValueError):
    """An order the shop will not make."""


def make_drink(
    base: str,
    toppings: Iterable[str] = (),
    *,
    loyalty: bool = False,
    bases: Mapping[str, Callable[[], Beverage]] = BASES,
    menu: Mapping[str, type[Topping]] = TOPPINGS,
) -> Beverage:
    """Build a drink from the names on an order.

    Names are looked up in read-only tables, never turned into attribute or
    class names, so an order can only ever produce something on the menu.
    The discount, when given, is applied last so that it covers every topping.
    """
    requested = [str(name).strip().lower() for name in toppings]
    base_key = str(base).strip().lower()
    if base_key not in bases:
        raise OrderError(f"{base!r} is not on the menu")
    unknown = [name for name in requested if name not in menu]
    if unknown:
        raise OrderError(f"not on the menu: {', '.join(map(repr, unknown))}")
    if len(requested) > MAX_TOPPINGS:
        raise OrderError(f"at most {MAX_TOPPINGS} toppings per drink, {len(requested)} requested")
    too_many = [name for name, count in Counter(requested).items() if count > MAX_EACH]
    if too_many:
        raise OrderError(f"at most {MAX_EACH} of each topping: {', '.join(too_many)}")

    drink = bases[base_key]()
    for name in requested:
        drink = menu[name](drink)
    if loyalty:
        drink = LoyaltyDiscount(drink)
    return drink


# --- Python's decorator syntax, for comparison --------------------------------

P = ParamSpec("P")
R = TypeVar("R")


def audited(log: list[str]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A Python function decorator that records every call and its outcome.

    This is the language feature written with ``@``, not the design pattern:
    it wraps a function once, when it is defined, whereas the pattern wraps
    objects one at a time while the program runs.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                log.append(f"{function.__name__} refused: {error}")
                raise
            log.append(f"{function.__name__} -> {result!r}")
            return result

        return wrapper

    return decorate


def main(orders: Iterable[tuple[str, list[str]]] | None = None) -> list[str]:
    """Print and return a receipt line for each order."""
    if orders is None:
        orders = [
            ("coffee", []),
            ("coffee", ["milk", "sugar"]),
            ("large coffee", ["oat milk", "vanilla syrup", "sugar", "sugar"]),
            ("espresso", ["extra shot", "whipped cream"]),
        ]
    lines = [make_drink(base, toppings).receipt_line() for base, toppings in orders]
    for line in lines:
        print(line)
    return lines


if __name__ == "__main__":
    main()
