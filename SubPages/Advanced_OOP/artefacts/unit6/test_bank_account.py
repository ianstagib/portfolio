"""Tests for the Unit 6 thread-safe bank account. Run from this folder with:

    python -m unittest -v test_bank_account

Concurrency bugs depend on how threads happen to be scheduled, so a test
that simply starts some threads and checks the result can pass by luck. The
race and deadlock tests therefore use a ``threading.Barrier`` to force the
harmful interleaving, which makes each failure certain rather than likely.
The stress tests then check the fixed classes under heavy load with Python
told to switch threads as often as it can. Every thread is joined with a
timeout, so a deadlock fails the test instead of hanging the suite. The
doctests in bank_account.py are collected too (see load_tests).
"""

import doctest
import contextlib
import io
import sys
import threading
import time
import unittest
from decimal import Decimal

import bank_account
from bank_account import (
    BankAccount,
    DeadlockDetected,
    InsufficientFunds,
    TellerDesk,
    Transaction,
    UnsafeAccount,
    main,
    naive_transfer,
    to_amount,
)

JOIN_TIMEOUT = 10.0


def run_threads(*targets):
    """Run each target in its own thread and return the exceptions raised.

    Fails loudly if any thread is still alive after JOIN_TIMEOUT, which is
    how a deadlock shows itself.
    """
    errors = []
    errors_lock = threading.Lock()

    def guarded(target):
        try:
            target()
        except BaseException as error:
            with errors_lock:
                errors.append(error)

    threads = [threading.Thread(target=guarded, args=(t,), daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
    alive = [thread for thread in threads if thread.is_alive()]
    if alive:
        raise AssertionError(f"{len(alive)} thread(s) still running: deadlock?")
    return errors


def repeat(times, action):
    def run():
        for _ in range(times):
            action()
    return run


@contextlib.contextmanager
def frequent_switching():
    """Ask the interpreter to switch threads as often as it can."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


class RaceConditionTests(unittest.TestCase):
    """The unsafe account loses money and overdraws; the safe one does not."""

    def test_unsafe_deposits_lose_an_update_when_interleaved(self):
        barrier = threading.Barrier(2, timeout=2)
        account = UnsafeAccount(0, pause=barrier.wait)
        run_threads(lambda: account.deposit(100), lambda: account.deposit(100))
        # Both threads read 0 before either wrote, so one deposit vanished.
        self.assertEqual(account.balance, Decimal("100.00"))

    def test_unsafe_withdrawals_overdraw_when_interleaved(self):
        barrier = threading.Barrier(2, timeout=2)
        account = UnsafeAccount(100, pause=barrier.wait)
        errors = run_threads(lambda: account.withdraw(100), lambda: account.withdraw(100))
        # Time of check to time of use: both passed the check, both debited.
        self.assertEqual(errors, [])
        self.assertEqual(account.balance, Decimal("-100.00"))

    def test_unsafe_account_loses_deposits_under_load(self):
        account = UnsafeAccount()
        run_threads(*[repeat(500, lambda: account.deposit(1)) for _ in range(8)])
        self.assertLess(account.balance, Decimal(4000))

    def test_safe_account_keeps_every_deposit_under_load(self):
        account = BankAccount("Alice")
        with frequent_switching():
            run_threads(*[repeat(1000, lambda: account.deposit(1)) for _ in range(8)])
        self.assertEqual(account.balance, Decimal("8000.00"))
        self.assertEqual(len(account.history()), 8000)

    def test_safe_account_cannot_be_overdrawn_by_concurrent_withdrawals(self):
        account = BankAccount("Bob", 100)
        with frequent_switching():
            errors = run_threads(*[lambda: account.withdraw(10) for _ in range(50)])
        self.assertEqual(len(errors), 40)
        self.assertTrue(all(isinstance(e, InsufficientFunds) for e in errors))
        self.assertEqual(account.balance, Decimal("0.00"))

    def test_mixed_deposits_and_withdrawals_balance_exactly(self):
        account = BankAccount("Carol", 1000)
        refused = []
        refused_lock = threading.Lock()

        def withdraw_some():
            for _ in range(200):
                try:
                    account.withdraw(3)
                except InsufficientFunds:
                    with refused_lock:
                        refused.append(1)

        with frequent_switching():
            run_threads(*[repeat(200, lambda: account.deposit(2)) for _ in range(4)],
                        *[withdraw_some for _ in range(4)])
        withdrawn = 3 * (800 - len(refused))
        self.assertEqual(account.balance, Decimal(1000 + 2 * 800 - withdrawn))
        self.assertGreaterEqual(account.balance, 0)
        # The history replays to the same balance, so no entry was lost.
        self.assertEqual(sum(t.amount for t in account.history()), account.balance)

    def test_balance_never_seen_negative_by_a_reader(self):
        account = BankAccount("Dan", 50)
        seen = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                seen.append(account.balance)

        def spender():
            for _ in range(500):
                with contextlib.suppress(InsufficientFunds):
                    account.withdraw(7)
                account.deposit(5)
            stop.set()

        with frequent_switching():
            run_threads(reader, spender, spender)
        self.assertTrue(seen)
        self.assertGreaterEqual(min(seen), 0)


class DeadlockTests(unittest.TestCase):
    """Opposite transfers deadlock with inconsistent lock order, not with ranking."""

    def test_naive_opposite_transfers_deadlock_when_interleaved(self):
        a, b = BankAccount("A", 100), BankAccount("B", 100)
        barrier = threading.Barrier(2, timeout=2)
        errors = run_threads(
            lambda: naive_transfer(a, b, 10, after_first_lock=barrier.wait, timeout=0.3),
            lambda: naive_transfer(b, a, 10, after_first_lock=barrier.wait, timeout=0.3),
        )
        # Each thread held one lock and timed out waiting for the other.
        self.assertTrue(errors)
        self.assertTrue(all(isinstance(e, DeadlockDetected) for e in errors))
        # Giving up released the locks: the accounts still work.
        a.transfer_to(b, 1)
        self.assertEqual(a.balance + b.balance, Decimal(200))

    def test_ordered_opposite_transfers_complete(self):
        # Each account can afford all its transfers, so none is refused.
        a, b = BankAccount("A", 2000), BankAccount("B", 2000)
        with frequent_switching():
            errors = run_threads(repeat(2000, lambda: a.transfer_to(b, 1)),
                                 repeat(2000, lambda: b.transfer_to(a, 1)))
        self.assertEqual(errors, [])
        self.assertEqual(a.balance + b.balance, Decimal(4000))

    def test_ring_of_transfers_like_dining_philosophers_completes(self):
        # Five accounts in a ring, each thread paying its neighbour: the
        # dining philosophers set-up that deadlocks with left-then-right locking.
        accounts = [BankAccount(f"P{i}", 500) for i in range(5)]
        targets = [repeat(500, lambda i=i: accounts[i].transfer_to(accounts[(i + 1) % 5], 1))
                   for i in range(5)]
        with frequent_switching():
            errors = run_threads(*targets)
        self.assertEqual(errors, [])
        self.assertEqual(sum(a.balance for a in accounts), Decimal(2500))

    def test_lock_order_is_by_account_number_not_by_direction(self):
        a, b = BankAccount("A", 10), BankAccount("B", 10)
        order = []

        class Recording:
            def __init__(self, name, lock):
                self.name, self.lock = name, lock
            def __enter__(self):
                order.append(self.name)
                return self.lock.__enter__()
            def __exit__(self, *exc):
                return self.lock.__exit__(*exc)

        a._lock, b._lock = Recording("a", a._lock), Recording("b", b._lock)
        a.transfer_to(b, 1)
        b.transfer_to(a, 1)
        self.assertLess(a.number, b.number)
        self.assertEqual(order, ["a", "b", "a", "b"])

    def test_transfer_to_self_is_refused(self):
        # With a non-reentrant Lock this would make the thread wait for itself.
        account = BankAccount("Self", 10)
        with self.assertRaises(ValueError):
            account.transfer_to(account, 1)

    def test_failed_transfer_changes_neither_account(self):
        a, b = BankAccount("A", 5), BankAccount("B", 5)
        with self.assertRaises(InsufficientFunds):
            a.transfer_to(b, 6)
        self.assertEqual((a.balance, b.balance), (Decimal(5), Decimal(5)))
        self.assertEqual(len(a.history()), 1)


class MonitorTests(unittest.TestCase):
    """withdraw_when_funded waits on a condition variable that shares the lock."""

    def test_waiting_withdrawal_completes_after_a_deposit(self):
        account = BankAccount("Erin")
        results = []
        waiter = threading.Thread(
            target=lambda: results.append(account.withdraw_when_funded(30, timeout=5)))
        waiter.start()
        time.sleep(0.05)
        self.assertEqual(results, [])  # still waiting, and not holding the lock
        account.deposit(50)
        waiter.join(JOIN_TIMEOUT)
        self.assertEqual(results, [Decimal("20.00")])

    def test_waiting_withdrawal_times_out_without_funds(self):
        account = BankAccount("Finn", 10)
        start = time.perf_counter()
        with self.assertRaises(InsufficientFunds):
            account.withdraw_when_funded(20, timeout=0.1)
        self.assertLess(time.perf_counter() - start, 2)
        self.assertEqual(account.balance, Decimal(10))

    def test_one_deposit_releases_only_the_waiters_it_can_pay(self):
        account = BankAccount("Gail")
        ready = threading.Barrier(4, timeout=2)

        def wait_for_fifty():
            ready.wait()
            account.withdraw_when_funded(50, timeout=0.5)

        def deposit_hundred():
            ready.wait()
            time.sleep(0.05)
            account.deposit(100)

        errors = run_threads(wait_for_fifty, wait_for_fifty, wait_for_fifty, deposit_hundred)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InsufficientFunds)
        self.assertEqual(account.balance, Decimal(0))

    def test_transfer_wakes_a_waiter_on_the_receiving_account(self):
        payer, payee = BankAccount("H", 100), BankAccount("I")
        results = []
        waiter = threading.Thread(
            target=lambda: results.append(payee.withdraw_when_funded(40, timeout=5)))
        waiter.start()
        time.sleep(0.05)
        payer.transfer_to(payee, 40)
        waiter.join(JOIN_TIMEOUT)
        self.assertEqual(results, [Decimal(0)])


class SemaphoreTests(unittest.TestCase):
    """The bounded semaphore admits n threads at once; a lock admits one."""

    def test_no_more_than_the_tellers_serve_at_once(self):
        desk = TellerDesk(tellers=3)
        run_threads(*[lambda: desk.serve(lambda: time.sleep(0.02)) for _ in range(15)])
        self.assertEqual(desk.peak, 3)

    def test_one_teller_behaves_like_a_lock(self):
        desk = TellerDesk(tellers=1)
        run_threads(*[lambda: desk.serve(lambda: time.sleep(0.01)) for _ in range(6)])
        self.assertEqual(desk.peak, 1)

    def test_a_failing_task_gives_its_teller_back(self):
        desk = TellerDesk(tellers=1)

        def fail():
            raise RuntimeError("card declined")

        with self.assertRaises(RuntimeError):
            desk.serve(fail)
        self.assertEqual(desk.serve(lambda: "served"), "served")

    def test_a_desk_needs_a_teller(self):
        with self.assertRaises(ValueError):
            TellerDesk(tellers=0)


class EncapsulationTests(unittest.TestCase):
    """The lock and the data it guards are kept inside one object."""

    def test_balance_is_read_only(self):
        account = BankAccount("J", 10)
        with self.assertRaises(AttributeError):
            account.balance = Decimal(1_000_000)

    def test_no_public_attribute_exposes_the_lock(self):
        account = BankAccount("K")
        public = [name for name in dir(account) if not name.startswith("_")]
        lock_types = (type(threading.Lock()), threading.Condition)
        self.assertFalse([n for n in public if isinstance(getattr(account, n), lock_types)])

    def test_history_is_an_immutable_snapshot(self):
        account = BankAccount("L", 10)
        snapshot = account.history()
        account.deposit(5)
        self.assertEqual(len(snapshot), 1)
        self.assertIsInstance(snapshot, tuple)
        with self.assertRaises(AttributeError):
            snapshot[0].amount = Decimal(999)  # frozen dataclass
        self.assertIsInstance(snapshot[0], Transaction)

    def test_account_numbers_are_unique_when_created_concurrently(self):
        made = []
        made_lock = threading.Lock()

        def open_accounts():
            for _ in range(200):
                account = BankAccount("M")
                with made_lock:
                    made.append(account.number)

        with frequent_switching():
            run_threads(*[open_accounts for _ in range(5)])
        self.assertEqual(len(set(made)), 1000)

    def test_amounts_are_validated(self):
        account = BankAccount("N", 10)
        for bad in (0.1, True, 0, -5, "NaN", "Infinity", "0.001", "ten"):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                account.deposit(bad)
        self.assertEqual(to_amount("2.5"), Decimal("2.50"))
        self.assertEqual(account.balance, Decimal(10))


class LimitTests(unittest.TestCase):
    """What the design does not achieve, recorded so it stays visible."""

    def test_a_lock_does_not_lock_the_data(self):
        # Python privacy is a convention: code that ignores it can bypass the
        # lock, exactly as Nguyen (2022, ch. 14) warns.
        account = BankAccount("O", 10)
        account._balance = Decimal(1_000_000)
        self.assertEqual(account.balance, Decimal(1_000_000))

    def test_ranked_locking_serialises_transfers_between_the_same_pair(self):
        # The fix removes the deadlock but not the waiting: two threads moving
        # money between the same two accounts take turns.
        a, b = BankAccount("P", 10), BankAccount("Q", 10)
        inside = []
        ready = threading.Event()
        original = a._debit

        def slow_debit(value, kind):
            inside.append(threading.get_ident())
            ready.set()
            time.sleep(0.1)
            original(value, kind)

        a._debit = slow_debit
        first = threading.Thread(target=lambda: a.transfer_to(b, 1))
        first.start()
        ready.wait(2)
        start = time.perf_counter()
        b.transfer_to(a, 1)  # must wait for the first transfer to finish
        waited = time.perf_counter() - start
        first.join(JOIN_TIMEOUT)
        self.assertGreater(waited, 0.05)


class DemonstrationTests(unittest.TestCase):
    def test_main_shows_each_problem_and_fix(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main()
        text = output.getvalue()
        self.assertIn("BankAccount    balance  4000.00", text)
        self.assertIn("UnsafeAccount  balance  -100.00", text)
        self.assertIn("deadlocks detected", text)
        self.assertIn("total 1000.00", text)
        self.assertIn("most served at once 3", text)


def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(bank_account))
    return tests


if __name__ == "__main__":
    unittest.main()
