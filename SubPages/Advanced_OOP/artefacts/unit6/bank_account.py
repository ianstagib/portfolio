"""Unit 6 exercise: a thread-safe bank account in Python.

The seminar asks for a small multi-threaded application, such as a bank
account with concurrent deposits and withdrawals, made thread safe with
threading locks. This module contains both the broken and the fixed
versions so that each problem can be shown and then removed.

* ``UnsafeAccount`` has no lock. Its deposit reads the balance, pauses and
  writes it back, so concurrent deposits lose updates (a race condition),
  and its withdrawal checks the balance before debiting it, so two
  withdrawals can both pass the check and overdraw the account (a
  time-of-check to time-of-use bug).
* ``BankAccount`` is a monitor: the balance and the lock that guards it are
  private to one object, and every public method takes the lock, so callers
  cannot forget to. A condition variable on the same lock lets a withdrawal
  wait until a deposit makes it affordable.
* ``naive_transfer`` locks the source account and then the destination, so
  two opposite transfers can each hold one lock and wait for the other (a
  deadlock). ``BankAccount.transfer_to`` removes the circular wait by always
  taking the two locks in account-number order.
* ``TellerDesk`` uses a bounded semaphore to let at most n operations run at
  once, the case where a lock (n = 1) is too strict.

Only the standard library is used. Run the demonstration with
``python bank_account.py`` and the tests with
``python -m unittest -v test_bank_account``.

>>> account = BankAccount("Alice", "100.00")
>>> account.deposit("25.50")
Decimal('125.50')
>>> account.withdraw(200)  # doctest: +ELLIPSIS
Traceback (most recent call last):
...
bank_account.InsufficientFunds: account ... holds 125.50, cannot withdraw 200.00
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TypeVar

__all__ = [
    "BankAccount",
    "DeadlockDetected",
    "InsufficientFunds",
    "TellerDesk",
    "Transaction",
    "UnsafeAccount",
    "count_primes",
    "main",
    "naive_transfer",
    "to_amount",
]

T = TypeVar("T")
PENNY = Decimal("0.01")
AmountLike = Decimal | int | str


class InsufficientFunds(Exception):
    """Raised when a withdrawal or transfer would overdraw an account."""


class DeadlockDetected(RuntimeError):
    """Raised by ``naive_transfer`` when it gives up waiting for a lock.

    A genuinely naive transfer would wait for ever; the timeout exists only so
    that the demonstration and the tests finish.
    """


def to_amount(value: AmountLike) -> Decimal:
    """Convert ``value`` to a positive amount in pounds and pence.

    Floats are refused because binary fractions cannot hold most pence
    exactly, and ``bool`` is refused although Python treats it as an int.

    >>> to_amount("10.5")
    Decimal('10.50')
    >>> to_amount(0.1)
    Traceback (most recent call last):
    ...
    TypeError: amount must be a Decimal, int or str, not float
    """
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise TypeError(
            f"amount must be a Decimal, int or str, not {type(value).__name__}"
        )
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"not a number: {value!r}") from None
    if not amount.is_finite() or amount <= 0:
        raise ValueError(f"amount must be positive and finite, not {value!r}")
    if amount != amount.quantize(PENNY):
        raise ValueError(f"amount has fractions of a penny: {value!r}")
    return amount.quantize(PENNY)


def _yield_thread() -> None:
    """Give other threads a chance to run, as slow work would."""
    time.sleep(0)


# ---------------------------------------------------------------------------
# The broken account: shared state with no synchronisation
# ---------------------------------------------------------------------------


class UnsafeAccount:
    """A bank account with no lock, kept to show what goes wrong.

    ``pause`` runs between reading and writing the balance. By default it
    yields to other threads, standing in for the slow work (a database call,
    a fraud check) that real code does there. Tests pass a barrier instead,
    which forces two threads to interleave and makes the race repeatable.
    """

    def __init__(
        self, opening: AmountLike = 0, pause: Callable[[], object] = _yield_thread
    ) -> None:
        self.balance = Decimal(opening).quantize(PENNY)
        self._pause = pause

    def deposit(self, amount: AmountLike) -> None:
        """Read, pause, write: a second deposit in the pause is lost."""
        value = to_amount(amount)
        current = self.balance
        self._pause()
        self.balance = current + value

    def withdraw(self, amount: AmountLike) -> None:
        """Check, pause, debit: two withdrawals can both pass the check."""
        value = to_amount(amount)
        if self.balance < value:
            raise InsufficientFunds(f"balance {self.balance} below {value}")
        self._pause()
        self.balance -= value


# ---------------------------------------------------------------------------
# The thread-safe account: a monitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transaction:
    """One entry in an account's history. Frozen, so it is safe to share."""

    kind: str
    amount: Decimal
    balance_after: Decimal


class BankAccount:
    """A thread-safe account in the style of a monitor.

    The balance, the history and the lock live together in one object and
    none of them is public. Every public method acquires the lock, so the
    check and the update in a withdrawal happen as one step. The lock is a
    plain ``threading.Lock``, which is not reentrant, so code that already
    holds it calls the private ``_credit`` and ``_debit`` helpers rather than
    the public methods.
    """

    _numbers = itertools.count(1)
    _numbers_lock = threading.Lock()

    def __init__(self, owner: str, opening: AmountLike = 0) -> None:
        with BankAccount._numbers_lock:
            self._number = next(BankAccount._numbers)
        self._owner = owner
        self._lock = threading.Lock()
        self._funds_added = threading.Condition(self._lock)
        self._balance = Decimal(0).quantize(PENNY)
        self._history: list[Transaction] = []
        if Decimal(opening) != 0:
            self._credit(to_amount(opening), "opening")

    def __repr__(self) -> str:
        return f"BankAccount({self._owner!r}, number={self._number})"

    @property
    def number(self) -> int:
        """Unique account number, also used to order lock acquisition."""
        return self._number

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def balance(self) -> Decimal:
        """The current balance, read under the lock."""
        with self._lock:
            return self._balance

    def history(self) -> tuple[Transaction, ...]:
        """A snapshot of the history, copied under the lock."""
        with self._lock:
            return tuple(self._history)

    def deposit(self, amount: AmountLike) -> Decimal:
        """Add ``amount`` and wake any withdrawal waiting for funds."""
        value = to_amount(amount)
        with self._lock:
            self._credit(value, "deposit")
            self._funds_added.notify_all()
            return self._balance

    def withdraw(self, amount: AmountLike) -> Decimal:
        """Remove ``amount``, checking and debiting while holding the lock."""
        value = to_amount(amount)
        with self._lock:
            self._require(value)
            self._debit(value, "withdrawal")
            return self._balance

    def withdraw_when_funded(self, amount: AmountLike, timeout: float) -> Decimal:
        """Wait up to ``timeout`` seconds for the balance to cover ``amount``.

        This is the monitor's condition variable at work: the thread releases
        the lock while it waits, so depositors can get in, and re-checks the
        condition each time it is woken.
        """
        value = to_amount(amount)
        with self._funds_added:
            if not self._funds_added.wait_for(
                lambda: self._balance >= value, timeout=timeout
            ):
                raise InsufficientFunds(
                    f"account {self._number} did not reach {value} "
                    f"within {timeout} s"
                )
            self._debit(value, "withdrawal")
            return self._balance

    def transfer_to(self, other: BankAccount, amount: AmountLike) -> None:
        """Move ``amount`` from this account to ``other`` atomically.

        Both locks are needed. Taking them in account-number order, whichever
        direction the money moves, means no two transfers can each hold the
        lock the other wants, which removes the circular wait that a deadlock
        needs.
        """
        value = to_amount(amount)
        if other is self:
            # Lock is not reentrant: locking the same account twice would
            # leave this thread waiting for itself.
            raise ValueError("cannot transfer an account to itself")
        first, second = sorted((self, other), key=lambda account: account.number)
        with first._lock, second._lock:
            self._require(value)
            self._debit(value, f"transfer to {other.number}")
            other._credit(value, f"transfer from {self.number}")
            other._funds_added.notify_all()

    # The helpers below assume the caller already holds self._lock.

    def _require(self, value: Decimal) -> None:
        if self._balance < value:
            raise InsufficientFunds(
                f"account {self._number} holds {self._balance}, "
                f"cannot withdraw {value}"
            )

    def _credit(self, value: Decimal, kind: str) -> None:
        self._balance += value
        self._history.append(Transaction(kind, value, self._balance))

    def _debit(self, value: Decimal, kind: str) -> None:
        self._balance -= value
        self._history.append(Transaction(kind, -value, self._balance))


# ---------------------------------------------------------------------------
# The broken transfer: inconsistent lock order
# ---------------------------------------------------------------------------


def naive_transfer(
    source: BankAccount,
    target: BankAccount,
    amount: AmountLike,
    *,
    after_first_lock: Callable[[], object] = _yield_thread,
    timeout: float = 1.0,
) -> None:
    """Transfer by locking the source and then the target. Do not use.

    If one thread moves money from A to B while another moves money from B to
    A, each can take its first lock and then wait for ever for the second: all
    four Coffman conditions hold. ``after_first_lock`` lets the tests make that
    interleaving certain. Instead of hanging, this version raises
    ``DeadlockDetected`` after ``timeout`` seconds and releases what it holds.
    """
    value = to_amount(amount)
    with source._lock:
        after_first_lock()
        if not target._lock.acquire(timeout=timeout):
            raise DeadlockDetected(
                f"holding account {source.number}, waited {timeout} s "
                f"for account {target.number}"
            )
        try:
            source._require(value)
            source._debit(value, f"transfer to {target.number}")
            target._credit(value, f"transfer from {source.number}")
        finally:
            target._lock.release()


# ---------------------------------------------------------------------------
# A semaphore: at most n at once
# ---------------------------------------------------------------------------


class TellerDesk:
    """Serves tasks with at most ``tellers`` running at the same time.

    The bounded semaphore limits how many threads are inside ``serve``; it
    does not protect data. The busy count is shared, so it has its own lock.
    """

    def __init__(self, tellers: int) -> None:
        if tellers < 1:
            raise ValueError("a desk needs at least one teller")
        self._tellers = threading.BoundedSemaphore(tellers)
        self._count_lock = threading.Lock()
        self._busy = 0
        self._peak = 0

    @property
    def peak(self) -> int:
        """The most tasks that were ever being served at once."""
        with self._count_lock:
            return self._peak

    def serve(self, task: Callable[[], T]) -> T:
        with self._tellers:
            with self._count_lock:
                self._busy += 1
                self._peak = max(self._peak, self._busy)
            try:
                return task()
            finally:
                with self._count_lock:
                    self._busy -= 1


# ---------------------------------------------------------------------------
# CPU-bound work for the threads versus processes comparison in the notebook
# ---------------------------------------------------------------------------


def count_primes(limit: int) -> int:
    """Count primes below ``limit`` by trial division: deliberately slow.

    >>> count_primes(30)
    10
    """
    count = 0
    for candidate in range(2, limit):
        divisor = 2
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                break
            divisor += 1
        else:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


def _repeat(times: int, action: Callable[[], object]) -> Callable[[], None]:
    """A thread target that calls ``action`` ``times`` times."""

    def run() -> None:
        for _ in range(times):
            action()

    return run


def _run_together(*targets: Callable[[], object]) -> list[BaseException]:
    """Start one thread per target, wait for all, return what they raised."""
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def guarded(target: Callable[[], object]) -> None:
        try:
            target()
        except BaseException as error:  # reported to the caller
            with errors_lock:
                errors.append(error)

    threads = [threading.Thread(target=guarded, args=(t,)) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def main() -> None:
    """Show each problem and its fix."""
    threads, deposits = 8, 500
    expected = Decimal(threads * deposits)

    unsafe = UnsafeAccount()
    _run_together(*[_repeat(deposits, lambda: unsafe.deposit(1)) for _ in range(threads)])
    print(f"1. Race condition: {threads} threads x {deposits} deposits of 1.00")
    print(f"   UnsafeAccount  balance {unsafe.balance:>8}  expected {expected}")
    safe = BankAccount("Alice")
    _run_together(*[_repeat(deposits, lambda: safe.deposit(1)) for _ in range(threads)])
    print(f"   BankAccount    balance {safe.balance:>8}  expected {expected}")

    barrier = threading.Barrier(2)
    unsafe = UnsafeAccount(100, pause=barrier.wait)
    errors = _run_together(lambda: unsafe.withdraw(100), lambda: unsafe.withdraw(100))
    print("2. Time of check to time of use: two withdrawals of 100 from 100")
    print(f"   UnsafeAccount  balance {unsafe.balance:>8}  refused {len(errors)}")
    safe = BankAccount("Bob", 100)
    errors = _run_together(lambda: safe.withdraw(100), lambda: safe.withdraw(100))
    print(f"   BankAccount    balance {safe.balance:>8}  refused {len(errors)}")

    a, b = BankAccount("Carol", 500), BankAccount("Dan", 500)
    barrier = threading.Barrier(2)
    errors = _run_together(
        lambda: naive_transfer(a, b, 10, after_first_lock=barrier.wait, timeout=0.5),
        lambda: naive_transfer(b, a, 10, after_first_lock=barrier.wait, timeout=0.5),
    )
    print("3. Deadlock: opposite transfers between two accounts")
    print(f"   naive_transfer           deadlocks detected {len(errors)}")
    start = time.perf_counter()
    _run_together(
        _repeat(1000, lambda: a.transfer_to(b, 1)),
        _repeat(1000, lambda: b.transfer_to(a, 1)),
    )
    print(
        f"   transfer_to (ordered)    2 x 1000 transfers in "
        f"{time.perf_counter() - start:.2f} s, total {a.balance + b.balance}"
    )

    waiting = BankAccount("Erin")
    result: list[Decimal] = []
    waiter = threading.Thread(
        target=lambda: result.append(waiting.withdraw_when_funded(30, timeout=2))
    )
    waiter.start()
    time.sleep(0.05)
    waiting.deposit(50)
    waiter.join()
    print("4. Monitor: a withdrawal of 30 waits for a deposit of 50")
    print(f"   balance after the waiting withdrawal {result[0]}")

    desk = TellerDesk(tellers=3)
    _run_together(*[lambda: desk.serve(lambda: time.sleep(0.02)) for _ in range(12)])
    print("5. Semaphore: 12 customers, 3 tellers")
    print(f"   most served at once {desk.peak}")


if __name__ == "__main__":
    main()
