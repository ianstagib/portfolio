"""Tests for the Unit 2 case study. Run from this folder with:

    python -m unittest -v test_shop

The first group pins down what is wrong with shop_initial.py. The rest test
shop_solid.py, one group per SOLID principle. The doctests in both modules
are collected too (see load_tests at the end).
"""

import contextlib
import doctest
import inspect
import io
import unittest
from decimal import Decimal

import shop_initial
import shop_solid
from shop_solid import (
    CreditCardPayment,
    CryptoPayment,
    Discount,
    FixedAmountDiscount,
    NoDiscount,
    Order,
    PaymentMethod,
    PaymentProcessor,
    PaymentResult,
    PayPalPayment,
    PercentageDiscount,
    Product,
    Refundable,
    RefundService,
)

CARD = "4111 1111 1111 1111"
NOTEBOOK = Product("NB-01", "Notebook", "4.50")
PEN = Product("PN-02", "Pen", "1.20")


def make_order():
    order = Order()
    order.add_item(NOTEBOOK, 2)
    order.add_item(PEN)
    return order  # subtotal 10.20


def every_payment_method():
    """One instance of every concrete PaymentMethod, so the contract tests
    cover a subclass as soon as it is written."""
    return [
        CreditCardPayment(CARD),
        PayPalPayment("sam@example.com"),
        CryptoPayment("0xA1B2C3D4E5", "50000"),
    ]


class InitialDesignProblems(unittest.TestCase):
    """What the original code does wrong. These pass because the faults are real."""

    def test_order_has_several_reasons_to_change(self):  # SRP
        public = {name for name, _ in inspect.getmembers(shop_initial.Order, inspect.isfunction)
                  if not name.startswith("_")}
        self.assertEqual(public, {"add_item", "apply_discount", "calculate_total", "pay"})

    def test_payment_types_are_hard_coded_in_order(self):  # OCP, DIP
        source = inspect.getsource(shop_initial.Order.pay)
        self.assertIn('"credit"', source)
        self.assertIn('"paypal"', source)

    def test_pay_only_prints_and_returns_nothing(self):
        order = shop_initial.Order()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = order.pay("credit")
        self.assertIsNone(result)
        self.assertIn("credit card", out.getvalue())

    def test_unknown_payment_type_is_silently_ignored(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = shop_initial.Order().pay("applepay")
        self.assertIsNone(result)
        self.assertEqual(out.getvalue(), "")

    def test_float_prices_give_rounding_errors(self):
        order = shop_initial.Order()
        order.add_item(shop_initial.Item("Notebook", 0.10))
        order.add_item(shop_initial.Item("Pencil", 0.20))
        self.assertNotEqual(order.calculate_total(), 0.30)

    def test_fixed_discount_can_make_total_negative(self):
        order = shop_initial.Order()
        order.add_item(shop_initial.Item("Pencil", 2.00))
        order.apply_discount("fixed", 5)
        self.assertLess(order.calculate_total(), 0)


class SingleResponsibilityTests(unittest.TestCase):

    def test_order_knows_nothing_about_payment(self):
        self.assertFalse([name for name in dir(Order) if "pay" in name.lower()])

    def test_subtotal_counts_quantities(self):
        self.assertEqual(make_order().subtotal(), Decimal("10.20"))

    def test_adding_same_product_increases_quantity(self):
        order = make_order()
        order.add_item(PEN, 3)
        self.assertEqual(dict((p.sku, q) for p, q in order.items)["PN-02"], 4)

    def test_same_sku_different_product_is_refused(self):
        with self.assertRaises(ValueError):
            make_order().add_item(Product("PN-02", "Pen", "9.99"))

    def test_items_view_cannot_change_the_order(self):
        order = make_order()
        with self.assertRaises(AttributeError):
            order.items.append((PEN, 1))

    def test_remove_item(self):
        order = make_order()
        order.remove_item("NB-01")
        self.assertEqual(order.subtotal(), Decimal("1.20"))
        with self.assertRaises(KeyError):
            order.remove_item("NB-01")

    def test_money_is_exact(self):
        order = Order()
        order.add_item(Product("NB-01", "Notebook", "0.10"))
        order.add_item(Product("PC-03", "Pencil", "0.20"))
        self.assertEqual(order.subtotal(), Decimal("0.30"))

    def test_invalid_products_and_quantities_are_refused(self):
        for bad in (lambda: Product("X", "Free", "0"),
                    lambda: Product("X", "Float", 0.1),
                    lambda: Product("", "No SKU", "1"),
                    lambda: make_order().add_item(PEN, 0),
                    lambda: make_order().add_item(PEN, True)):
            with self.assertRaises((ValueError, TypeError)):
                bad()


class OpenClosedTests(unittest.TestCase):

    def test_discounts(self):
        order = make_order()
        self.assertEqual(order.total(), Decimal("10.20"))
        self.assertEqual(order.total(NoDiscount()), Decimal("10.20"))
        self.assertEqual(order.total(PercentageDiscount(10)), Decimal("9.18"))
        self.assertEqual(order.total(FixedAmountDiscount("2.50")), Decimal("7.70"))

    def test_fixed_discount_never_goes_below_zero(self):
        self.assertEqual(make_order().total(FixedAmountDiscount(50)), Decimal("0.00"))

    def test_new_discount_needs_no_change_to_order(self):
        class BuyOneGetOneFreePens(Discount):
            def __init__(self, order):
                self.order = order

            def amount_off(self, subtotal):
                pens = sum(q for p, q in self.order.items if p.sku == "PN-02")
                return PEN.price * (pens // 2)

        order = make_order()
        order.add_item(PEN)  # two pens now
        self.assertEqual(order.total(BuyOneGetOneFreePens(order)), Decimal("10.20"))

    def test_new_payment_method_needs_no_change_to_processor(self):
        class BankTransferPayment(PaymentMethod):
            description = "bank transfer"

            def _take_payment(self, amount):
                return self._result(True, amount, "transfer requested")

        result = PaymentProcessor(BankTransferPayment()).checkout(make_order())
        self.assertTrue(result.succeeded)
        self.assertEqual(result.amount, Decimal("10.20"))

    def test_processor_source_names_no_concrete_method(self):
        source = inspect.getsource(PaymentProcessor)
        for concrete in ("CreditCard", "PayPal", "Crypto", '"credit"', '"paypal"'):
            self.assertNotIn(concrete, source)


class LiskovSubstitutionTests(unittest.TestCase):
    """The same contract, run against every PaymentMethod subclass."""

    def test_any_positive_amount_gives_a_result_for_that_amount(self):
        for method in every_payment_method():
            for amount in (Decimal("0.01"), Decimal("10.20"), Decimal("4999.99")):
                with self.subTest(method=method.description, amount=amount):
                    result = method.pay(amount)
                    self.assertIsInstance(result, PaymentResult)
                    self.assertEqual(result.amount, amount)
                    self.assertTrue(result.reference)

    def test_invalid_amounts_are_refused_the_same_way(self):
        for method in every_payment_method():
            for amount in (Decimal("0"), Decimal("-1")):
                with self.subTest(method=method.description, amount=amount):
                    with self.assertRaises(ValueError):
                        method.pay(amount)

    def test_decline_is_a_result_not_an_exception(self):
        card = CreditCardPayment(CARD, limit="5.00")
        result = PaymentProcessor(card).checkout(make_order())
        self.assertFalse(result.succeeded)
        self.assertIn("declined", result.message)

    def test_processor_works_with_every_method(self):
        for method in every_payment_method():
            with self.subTest(method=method.description):
                result = PaymentProcessor(method).checkout(make_order(), PercentageDiscount(10))
                self.assertTrue(result.succeeded)
                self.assertEqual(result.amount, Decimal("9.18"))

    def test_faulty_discount_cannot_produce_negative_total(self):
        class Broken(Discount):
            def amount_off(self, subtotal):
                return subtotal + 1

        with self.assertRaises(ValueError):
            make_order().total(Broken())


class InterfaceSegregationTests(unittest.TestCase):

    def test_payment_method_interface_is_small(self):
        abstract = PaymentMethod.__abstractmethods__
        self.assertEqual(abstract, {"description", "_take_payment"})
        self.assertFalse(hasattr(PaymentMethod, "refund"))

    def test_only_methods_that_can_refund_are_refundable(self):
        card, paypal, crypto = every_payment_method()
        self.assertIsInstance(card, Refundable)
        self.assertIsInstance(paypal, Refundable)
        self.assertNotIsInstance(crypto, Refundable)

    def test_refund_round_trip(self):
        card = CreditCardPayment(CARD)
        paid = PaymentProcessor(card).checkout(make_order())
        refunds = RefundService(card)
        self.assertTrue(refunds.refund(paid.reference, "4.50").succeeded)
        self.assertFalse(refunds.refund(paid.reference, "6.00").succeeded)  # only 5.70 left
        self.assertFalse(refunds.refund("CRED-999999", "1.00").succeeded)

    def test_refund_service_refuses_a_method_that_cannot_refund(self):
        with self.assertRaises(TypeError):
            RefundService(CryptoPayment("0xA1B2C3D4E5", "50000"))


class DependencyInversionTests(unittest.TestCase):

    def test_processor_accepts_a_test_double(self):
        class RecordingPayment(PaymentMethod):
            description = "test double"

            def __init__(self):
                self.amounts = []

            def _take_payment(self, amount):
                self.amounts.append(amount)
                return self._result(True, amount)

        double = RecordingPayment()
        PaymentProcessor(double).checkout(make_order(), FixedAmountDiscount("0.20"))
        self.assertEqual(double.amounts, [Decimal("10.00")])

    def test_processor_refuses_a_non_payment_method(self):
        with self.assertRaises(TypeError):
            PaymentProcessor("credit")

    def test_fully_discounted_order_takes_no_payment(self):
        class Exploding(PaymentMethod):
            description = "should not be called"

            def _take_payment(self, amount):
                raise AssertionError("payment taken for a free order")

        result = PaymentProcessor(Exploding()).checkout(make_order(), PercentageDiscount(100))
        self.assertTrue(result.succeeded)
        self.assertEqual(result.amount, Decimal("0.00"))

    def test_empty_order_cannot_be_checked_out(self):
        with self.assertRaises(ValueError):
            PaymentProcessor(PayPalPayment("sam@example.com")).checkout(Order())


class SensitiveDataTests(unittest.TestCase):

    def test_card_number_is_not_kept_or_shown(self):
        card = CreditCardPayment(CARD)
        result = card.pay(Decimal("1.00"))
        self.assertEqual(result.method, "credit card ending 1111")
        self.assertNotIn("4111111111111111", repr(vars(card)) + repr(result))

    def test_bad_card_number_is_not_echoed_in_the_error(self):
        with self.assertRaises(ValueError) as caught:
            CreditCardPayment("4111-1111-1111-1111")
        self.assertNotIn("4111", str(caught.exception))

    def test_paypal_email_is_masked(self):
        self.assertEqual(PayPalPayment("sam@example.com").description, "PayPal (s***@example.com)")


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(shop_initial))
    tests.addTests(doctest.DocTestSuite(shop_solid))
    return tests


if __name__ == "__main__":
    unittest.main()
