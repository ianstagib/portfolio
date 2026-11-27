"""Tests for the Collaborative Discussion 2 refactoring. Run from this folder with:

    python -m unittest -v test_payment_strategy

The first group checks that the refactored code behaves as the original did
for every valid input. The next groups check the claims made in the post:
the processor no longer branches on the method, a new method needs no edit,
and the method can change at run time. The last group checks the amount
validation the original lacked. The doctests in payment_strategy.py are
collected too (see load_tests).
"""

import ast
import contextlib
import doctest
import inspect
import io
import textwrap
import unittest
from decimal import Decimal

import payment_original
import payment_strategy
from payment_strategy import (
    PAYMENT_METHODS,
    BankTransferPayment,
    CreditCardPayment,
    PaymentProcessor,
    PaymentStrategy,
    PayPalPayment,
    to_amount,
)


def printed(call):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        call()
    return out.getvalue()


class CryptoPayment(PaymentStrategy):
    """A new method, defined only in this test file."""

    def __init__(self, wallet="bc1q-test-wallet"):
        self.wallet = wallet

    def pay(self, amount):
        return f"Processing crypto payment of ${amount} to {self.wallet}"


class SameBehaviourAsOriginal(unittest.TestCase):

    def test_same_output_for_every_method_and_amount(self):
        old = payment_original.PaymentProcessor()
        for payment_type in ("credit_card", "paypal", "bank_transfer"):
            for amount in (1, 100, "19.99", Decimal("250.50")):
                with self.subTest(payment_type=payment_type, amount=amount):
                    self.assertEqual(
                        printed(lambda: old.process_payment(payment_type, amount)),
                        printed(lambda: PaymentProcessor.for_type(payment_type)
                                .process_payment(amount)),
                    )

    def test_unknown_type_still_raises_the_same_error(self):
        for bad in ("crypto", "", "CREDIT_CARD", None, ["paypal"]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "Invalid payment type"):
                    PaymentProcessor.for_type(bad)
        with self.assertRaisesRegex(ValueError, "Invalid payment type"):
            payment_original.PaymentProcessor().process_payment("crypto", 1)

    def test_message_is_returned_as_well_as_printed(self):
        processor = PaymentProcessor(CreditCardPayment())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(processor.process_payment(5),
                             "Processing credit card payment of $5")


class NoBranchingInTheProcessor(unittest.TestCase):

    def test_processor_has_no_if_and_names_no_method(self):
        for method in (PaymentProcessor.process_payment, PaymentProcessor.for_type):
            body = ast.parse(textwrap.dedent(inspect.getsource(method)))
            self.assertFalse([n for n in ast.walk(body) if isinstance(n, (ast.If, ast.Match))])
        tree = ast.parse(textwrap.dedent(inspect.getsource(PaymentProcessor)))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for concrete in ("CreditCardPayment", "PayPalPayment", "BankTransferPayment"):
            self.assertNotIn(concrete, names)

    def test_original_branches_once_per_method(self):
        tree = ast.parse(inspect.getsource(payment_original))
        self.assertEqual(len([n for n in ast.walk(tree) if isinstance(n, ast.If)]), 3)

    def test_strategy_interface_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            PaymentStrategy()


class OpenForExtension(unittest.TestCase):

    def test_new_method_needs_no_edit(self):
        processor = PaymentProcessor(CryptoPayment())
        self.assertIn("to bc1q-test-wallet", printed(lambda: processor.process_payment(10)))

    def test_new_method_can_be_offered_by_name(self):
        methods = {**PAYMENT_METHODS, "crypto": CryptoPayment}
        processor = PaymentProcessor.for_type("crypto", methods)
        self.assertIsInstance(processor.strategy, CryptoPayment)
        with self.assertRaises(ValueError):
            PaymentProcessor.for_type("crypto")  # the default table is unchanged

    def test_default_table_is_read_only(self):
        with self.assertRaises(TypeError):
            PAYMENT_METHODS["crypto"] = CryptoPayment

    def test_strategy_can_change_at_run_time(self):
        processor = PaymentProcessor(CreditCardPayment())
        processor.strategy = BankTransferPayment()
        self.assertIn("bank transfer", printed(lambda: processor.process_payment(3)))

    def test_only_strategies_are_accepted(self):
        for bad in (None, "paypal", PayPalPayment, lambda amount: "paid"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                PaymentProcessor(bad)


class AmountValidation(unittest.TestCase):

    def test_good_amounts(self):
        self.assertEqual(to_amount(10), Decimal("10"))
        self.assertEqual(to_amount("0.01"), Decimal("0.01"))
        self.assertEqual(str(to_amount("1e2")), "100")

    def test_bad_amounts_never_reach_a_strategy(self):
        calls = []

        class Recording(PaymentStrategy):
            def pay(self, amount):
                calls.append(amount)
                return "paid"

        processor = PaymentProcessor(Recording())
        for bad in (0, -5, "abc", "NaN", "Infinity", "1.001", 0.1, True, None):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                processor.process_payment(bad)
        self.assertEqual(calls, [])

    def test_original_accepted_nonsense_amounts(self):
        out = printed(lambda: payment_original.PaymentProcessor()
                      .process_payment("paypal", -100))
        self.assertEqual(out, "Processing PayPal payment of $-100\n")


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(payment_strategy))
    return tests


if __name__ == "__main__":
    unittest.main()
