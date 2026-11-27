"""Collaborative Discussion 2: the PaymentProcessor refactored with Strategy.

The original ``process_payment()`` chose a payment method with an if/elif
chain, so every new method meant editing the processor. Here each method is
a strategy class and the processor only delegates to the one it holds:

* Strategy: ``PaymentStrategy``, an abstract class with ``pay()``.
* Concrete strategies: ``CreditCardPayment``, ``PayPalPayment`` and
  ``BankTransferPayment``.
* Context: ``PaymentProcessor``, which validates the amount and delegates.

Callers that still have a method name as text, for example from a web form,
use ``PaymentProcessor.for_type()``, which looks the name up in a read-only
table instead of branching on it.

>>> PaymentProcessor(PayPalPayment()).process_payment(25)
Processing PayPal payment of $25
'Processing PayPal payment of $25'
>>> PaymentProcessor.for_type("bank_transfer").process_payment("19.99")
Processing bank transfer of $19.99
'Processing bank transfer of $19.99'
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Union

__all__ = [
    "PaymentStrategy",
    "CreditCardPayment",
    "PayPalPayment",
    "BankTransferPayment",
    "PaymentProcessor",
    "PAYMENT_METHODS",
    "to_amount",
]

Amount = Union[int, str, Decimal]


def to_amount(value: Amount) -> Decimal:
    """Turn an amount into a Decimal, refusing anything a payment should not be.

    Floats are refused because binary fractions cannot hold most prices
    exactly (0.1 + 0.2 is not 0.3). Booleans, NaN, infinity, zero, negative
    amounts and fractions of a penny are refused too.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        raise TypeError("amount must be an int, a str or a Decimal, not a float")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"not an amount: {value!r}") from error
    if not amount.is_finite() or amount <= 0:
        raise ValueError("amount must be a positive, finite number")
    exponent = amount.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        raise ValueError("amount cannot include fractions of a penny")
    if isinstance(exponent, int) and exponent > 0:
        amount = amount.quantize(Decimal(1))  # show '1e2' as 100, not 1E+2
    return amount


# Step 1: the strategy interface


class PaymentStrategy(abc.ABC):
    """One way of taking a payment."""

    @abc.abstractmethod
    def pay(self, amount: Decimal) -> str:
        """Take the payment and describe what was done."""


# Step 2: concrete strategies


class CreditCardPayment(PaymentStrategy):
    def pay(self, amount: Decimal) -> str:
        return f"Processing credit card payment of ${amount}"


class PayPalPayment(PaymentStrategy):
    def pay(self, amount: Decimal) -> str:
        return f"Processing PayPal payment of ${amount}"


class BankTransferPayment(PaymentStrategy):
    def pay(self, amount: Decimal) -> str:
        return f"Processing bank transfer of ${amount}"


PAYMENT_METHODS: Mapping[str, type[PaymentStrategy]] = MappingProxyType(
    {
        "credit_card": CreditCardPayment,
        "paypal": PayPalPayment,
        "bank_transfer": BankTransferPayment,
    }
)


# Step 3: the processor delegates to whichever strategy it holds


class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy) -> None:
        self.strategy = strategy

    @property
    def strategy(self) -> PaymentStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: PaymentStrategy) -> None:
        if not isinstance(strategy, PaymentStrategy):
            raise TypeError("strategy must be a PaymentStrategy")
        self._strategy = strategy

    @classmethod
    def for_type(
        cls,
        payment_type: str,
        methods: Mapping[str, type[PaymentStrategy]] = PAYMENT_METHODS,
    ) -> PaymentProcessor:
        """Build a processor from a method name, as the original API took one."""
        try:
            strategy_class = methods[payment_type]
        except (KeyError, TypeError):
            raise ValueError("Invalid payment type") from None
        return cls(strategy_class())

    def process_payment(self, amount: Amount) -> str:
        message = self._strategy.pay(to_amount(amount))
        print(message)
        return message


# Step 4: usage


def main() -> None:
    processor = PaymentProcessor(CreditCardPayment())
    processor.process_payment(100)
    processor.strategy = PayPalPayment()  # swap the method at run time
    processor.process_payment("49.50")
    PaymentProcessor.for_type("bank_transfer").process_payment(Decimal("250"))


if __name__ == "__main__":
    main()
