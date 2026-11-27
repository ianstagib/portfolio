"""The Unit 2 shopping cart refactored to follow the SOLID principles.

Each class has one job, and the parts that vary (payment methods and
discounts) sit behind small abstractions:

* SRP: ``Order`` manages items and works out totals. ``PaymentProcessor``
  takes payments. Discounts and payment methods are their own classes.
* OCP: a new payment method or discount is a new class. Nothing that
  already exists is edited.
* LSP: every ``PaymentMethod`` keeps the same contract for ``pay()``. It
  accepts any positive amount, returns a ``PaymentResult`` and reports a
  decline in that result rather than by raising. The contract is tested
  against every subclass in test_shop.py.
* ISP: refunds are a separate ``Refundable`` interface, so a payment method
  that cannot refund (``CryptoPayment``) is not made to pretend it can.
* DIP: ``PaymentProcessor`` depends on the ``PaymentMethod`` and ``Discount``
  abstractions, which are passed in, and never names a concrete class.

Money is held as ``Decimal`` in pounds, rounded to the penny, so totals do
not pick up the float rounding errors that the original code had.

>>> order = Order()
>>> order.add_item(Product("NB-01", "Notebook", "0.10"))
>>> order.add_item(Product("PN-02", "Pencil", "0.20"))
>>> order.subtotal()
Decimal('0.30')
>>> processor = PaymentProcessor(CreditCardPayment("4111111111111111"))
>>> result = processor.checkout(order, PercentageDiscount(10))
>>> result.amount, result.method, result.succeeded
(Decimal('0.27'), 'credit card ending 1111', True)
"""

from __future__ import annotations

import abc
import itertools
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Product",
    "Order",
    "Discount",
    "NoDiscount",
    "PercentageDiscount",
    "FixedAmountDiscount",
    "PaymentResult",
    "PaymentMethod",
    "Refundable",
    "CreditCardPayment",
    "PayPalPayment",
    "CryptoPayment",
    "PaymentProcessor",
    "RefundService",
]

PENNY = Decimal("0.01")


def to_money(value: Decimal | int | str) -> Decimal:
    """Convert a value to pounds, rounded to the penny.

    Floats are refused, because ``Decimal(0.1)`` keeps the float's
    rounding error. Pass a string such as ``"0.10"`` instead.

    >>> to_money("19.999")
    Decimal('20.00')
    >>> to_money(0.1)
    Traceback (most recent call last):
    ...
    TypeError: use a str, int or Decimal for money, not float
    """
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"use a str, int or Decimal for money, not {type(value).__name__}")
    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"not a valid amount of money: {value!r}") from None
    if not amount.is_finite():
        raise ValueError("amount of money must be finite")
    return amount.quantize(PENNY, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# The cart (SRP: items and totals only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    """Something that can be sold. Frozen, so a price cannot change once the
    product is in a cart."""

    sku: str
    name: str
    price: Decimal

    def __post_init__(self) -> None:
        if not self.sku or not self.name:
            raise ValueError("a product needs a SKU and a name")
        price = to_money(self.price)
        if price <= 0:
            raise ValueError("price must be greater than zero")
        object.__setattr__(self, "price", price)


class Order:
    """Holds the products a customer wants and works out what they cost.

    It knows nothing about discounts beyond the ``Discount`` interface, and
    nothing about payment at all.
    """

    def __init__(self) -> None:
        self._lines: dict[str, list[Any]] = {}  # sku -> [product, quantity]

    def add_item(self, product: Product, quantity: int = 1) -> None:
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError("quantity must be a whole number of at least 1")
        line = self._lines.get(product.sku)
        if line is None:
            self._lines[product.sku] = [product, quantity]
        elif line[0] != product:
            raise ValueError(f"SKU {product.sku} is already in the order as a different product")
        else:
            line[1] += quantity

    def remove_item(self, sku: str) -> None:
        if sku not in self._lines:
            raise KeyError(sku)
        del self._lines[sku]

    @property
    def items(self) -> tuple[tuple[Product, int], ...]:
        """A read-only view of (product, quantity) pairs."""
        return tuple((product, quantity) for product, quantity in self._lines.values())

    def is_empty(self) -> bool:
        return not self._lines

    def subtotal(self) -> Decimal:
        return sum((product.price * quantity for product, quantity in self.items), Decimal("0.00"))

    def total(self, discount: Discount | None = None) -> Decimal:
        discount = discount or NoDiscount()
        subtotal = self.subtotal()
        reduction = to_money(discount.amount_off(subtotal))
        # Every discount promises 0 <= reduction <= subtotal. Checking it
        # here means a faulty discount cannot produce a negative total.
        if not Decimal("0") <= reduction <= subtotal:
            raise ValueError(f"{type(discount).__name__} broke its contract: {reduction} off {subtotal}")
        return subtotal - reduction


# ---------------------------------------------------------------------------
# Discounts (OCP: add a class, do not edit Order)
# ---------------------------------------------------------------------------


class Discount(abc.ABC):
    """Works out how much to take off a subtotal.

    Contract: ``amount_off()`` returns a value between zero and the subtotal.
    """

    @abc.abstractmethod
    def amount_off(self, subtotal: Decimal) -> Decimal:
        """Return the reduction for this subtotal."""


class NoDiscount(Discount):

    def amount_off(self, subtotal: Decimal) -> Decimal:
        return Decimal("0.00")


class PercentageDiscount(Discount):

    def __init__(self, percent: Decimal | int | str) -> None:
        percent = Decimal(str(percent)) if not isinstance(percent, Decimal) else percent
        if not Decimal("0") <= percent <= Decimal("100"):
            raise ValueError("percent must be between 0 and 100")
        self.percent = percent

    def amount_off(self, subtotal: Decimal) -> Decimal:
        return to_money(subtotal * self.percent / 100)


class FixedAmountDiscount(Discount):
    """Takes a fixed sum off, but never more than the subtotal."""

    def __init__(self, amount: Decimal | int | str) -> None:
        amount = to_money(amount)
        if amount < 0:
            raise ValueError("discount amount cannot be negative")
        self.amount = amount

    def amount_off(self, subtotal: Decimal) -> Decimal:
        return min(self.amount, subtotal)


# ---------------------------------------------------------------------------
# Payment methods (OCP, LSP, ISP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaymentResult:
    """What happened when a payment was attempted. Returned, never printed,
    so callers and tests can act on it."""

    succeeded: bool
    amount: Decimal
    method: str
    reference: str
    message: str = ""


class PaymentMethod(abc.ABC):
    """One way of paying. Kept to the single operation every method shares.

    Contract for ``pay(amount)``, which every subclass must keep (LSP):

    * accepts any amount greater than zero, already rounded to the penny,
      and adds no extra preconditions of its own;
    * returns a ``PaymentResult`` for that exact amount;
    * reports a decline as ``succeeded=False`` rather than raising.
      ``ValueError`` is raised only for an amount that breaks the
      precondition, and the base class checks that for every subclass.
    """

    _references = itertools.count(1)

    def pay(self, amount: Decimal) -> PaymentResult:
        """Template method: the base class checks the precondition once,
        so a subclass cannot weaken or strengthen it by accident."""
        amount = to_money(amount)
        if amount <= 0:
            raise ValueError("payment amount must be greater than zero")
        return self._take_payment(amount)

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A short, safe-to-log description, e.g. 'PayPal (a***@example.com)'."""

    @abc.abstractmethod
    def _take_payment(self, amount: Decimal) -> PaymentResult:
        """Do the work of taking ``amount``. Called only with a valid amount."""

    def _result(self, succeeded: bool, amount: Decimal, message: str = "") -> PaymentResult:
        prefix = type(self).__name__.removesuffix("Payment").upper()[:4]
        reference = f"{prefix}-{next(PaymentMethod._references):06d}"
        return PaymentResult(succeeded, amount, self.description, reference, message)


@runtime_checkable
class Refundable(Protocol):
    """A separate, optional role (ISP). Only methods that can give money back
    implement it, and only code that issues refunds depends on it."""

    def refund(self, reference: str, amount: Decimal) -> PaymentResult: ...


class _RefundablePayment(PaymentMethod):
    """Shared bookkeeping for the payment methods that can refund. It sits
    below ``PaymentMethod`` so that the base interface itself stays free of
    refunds; classes that cannot refund simply do not extend it."""

    def _ledger(self) -> dict[str, Decimal]:
        if not hasattr(self, "_taken"):
            self._taken: dict[str, Decimal] = {}
        return self._taken

    def _record(self, result: PaymentResult) -> PaymentResult:
        if result.succeeded:
            self._ledger()[result.reference] = result.amount
        return result

    def refund(self, reference: str, amount: Decimal) -> PaymentResult:
        amount = to_money(amount)
        taken = self._ledger()
        if reference not in taken:
            return self._result(False, amount, f"unknown payment reference {reference}")
        if not Decimal("0") < amount <= taken[reference]:
            return self._result(False, amount, "refund exceeds the amount still held")
        taken[reference] -= amount
        return self._result(True, amount, f"refund against {reference}")


class CreditCardPayment(_RefundablePayment):
    """Card payment. Only the last four digits are kept, so the full card
    number cannot leak into a result, a log line or an exception."""

    def __init__(self, card_number: str, limit: Decimal | int | str = "5000") -> None:
        digits = card_number.replace(" ", "")
        if not (digits.isdigit() and 12 <= len(digits) <= 19):
            raise ValueError("card number must be 12 to 19 digits")  # value left out on purpose
        self._last_four = digits[-4:]
        self._available = to_money(limit)

    @property
    def description(self) -> str:
        return f"credit card ending {self._last_four}"

    def _take_payment(self, amount: Decimal) -> PaymentResult:
        if amount > self._available:
            return self._result(False, amount, "declined: over the available limit")
        self._available -= amount
        return self._record(self._result(True, amount, "card payment authorised"))


class PayPalPayment(_RefundablePayment):

    def __init__(self, email: str) -> None:
        if "@" not in email:
            raise ValueError("a PayPal account needs an email address")
        self._email = email

    @property
    def description(self) -> str:
        user, _, domain = self._email.partition("@")
        return f"PayPal ({user[:1]}***@{domain})"

    def _take_payment(self, amount: Decimal) -> PaymentResult:
        return self._record(self._result(True, amount, "PayPal payment completed"))


class CryptoPayment(PaymentMethod):
    """Added after the first three classes, without editing any of them (OCP).

    It cannot refund, and does not implement ``Refundable`` (ISP). It converts
    pounds to a coin amount at a rate supplied when it is created; a real
    system would fetch the rate from an exchange.
    """

    def __init__(self, wallet: str, rate_gbp_per_coin: Decimal | int | str) -> None:
        if not wallet:
            raise ValueError("a crypto payment needs a wallet address")
        self._wallet = wallet
        self._rate = to_money(rate_gbp_per_coin)
        if self._rate <= 0:
            raise ValueError("exchange rate must be greater than zero")

    @property
    def description(self) -> str:
        return f"crypto wallet {self._wallet[:6]}..."

    def _take_payment(self, amount: Decimal) -> PaymentResult:
        coins = (amount / self._rate).quantize(Decimal("0.00000001"))
        return self._result(True, amount, f"sent {coins} coins")


# ---------------------------------------------------------------------------
# The high-level policy (DIP)
# ---------------------------------------------------------------------------


class PaymentProcessor:
    """Takes payment for an order.

    Depends only on the ``PaymentMethod`` and ``Discount`` abstractions. The
    concrete method is chosen by the caller and passed in (constructor
    injection), so this class never changes when a new method is added.
    """

    def __init__(self, method: PaymentMethod) -> None:
        if not isinstance(method, PaymentMethod):
            raise TypeError("method must be a PaymentMethod")
        self._method = method

    def checkout(self, order: Order, discount: Discount | None = None) -> PaymentResult:
        if order.is_empty():
            raise ValueError("cannot check out an empty order")
        total = order.total(discount)
        if total == 0:
            # A fully discounted order needs no payment at all.
            return PaymentResult(True, total, "no payment needed", "FREE")
        return self._method.pay(total)


class RefundService:
    """Issues refunds. Depends on ``Refundable`` only (ISP and DIP), so it can
    be given a card or PayPal account but not a crypto wallet.

    The check happens once, when the service is built, rather than as a type
    test inside ``PaymentProcessor`` every time a refund is asked for.
    """

    def __init__(self, method: Refundable) -> None:
        if not isinstance(method, Refundable):
            raise TypeError(f"{type(method).__name__} does not support refunds")
        self._method = method

    def refund(self, reference: str, amount: Decimal | int | str) -> PaymentResult:
        return self._method.refund(reference, to_money(amount))
