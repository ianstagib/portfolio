"""Tests for the Unit 4 Decorator exercise. Run from this folder with:

    python -m unittest -v test_coffee

The first group checks the brief: a base coffee whose class is left alone,
and toppings such as milk and sugar added at run time. The next groups check
what the pattern is meant to give: toppings that stack in any number and
order, a contract that every topping keeps, and a decorated drink that the
client treats exactly like a plain one. The order desk tests check that names
from an order can only ever produce something on the menu. The last group
records limits discussed in the reflection: a decorated drink is no longer
an instance of its base class, members a decorator adds are hidden by the
next layer, and a decorator that alters rather than adds makes the order of
layers matter. The doctests in coffee.py are collected too (see load_tests).
"""

import ast
import doctest
import inspect
import io
import itertools
import contextlib
import textwrap
import unittest

import coffee
from coffee import (
    BASES,
    MAX_EACH,
    MAX_TOPPINGS,
    TOPPINGS,
    Beverage,
    Coffee,
    Espresso,
    ExtraShot,
    LoyaltyDiscount,
    Milk,
    OatMilk,
    OrderError,
    Sugar,
    Topping,
    VanillaSyrup,
    WhippedCream,
    audited,
    format_price,
    make_drink,
)


class BriefRequirements(unittest.TestCase):
    """A base coffee with toppings such as milk and sugar added dynamically."""

    def test_base_coffee_has_a_description_and_a_price(self):
        self.assertEqual(Coffee().description(), "Coffee (regular)")
        self.assertEqual(Coffee().cost(), 250)

    def test_milk_and_sugar_wrap_a_coffee(self):
        drink = Sugar(Milk(Coffee()))
        self.assertEqual(drink.description(), "Coffee (regular), milk, sugar")
        self.assertEqual(drink.cost(), 250 + 40 + 15)

    def test_toppings_are_added_at_run_time_from_data(self):
        chosen = ["milk", "sugar", "sugar"]  # as if typed by a customer
        drink = Coffee()
        for name in chosen:
            drink = TOPPINGS[name](drink)
        self.assertEqual(drink.layers(), ("Coffee (regular)", "milk", "sugar", "sugar"))

    def test_coffee_class_knows_nothing_about_toppings(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(Coffee)))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {"Topping", *(cls.__name__ for cls in TOPPINGS.values())})

    def test_base_coffee_is_unchanged_by_wrapping(self):
        base = Coffee("small")
        Milk(base)
        Sugar(Sugar(base))
        self.assertEqual((base.description(), base.cost()), ("Coffee (small)", 220))

    def test_demonstration_prints_receipts(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            lines = coffee.main()
        self.assertEqual(len(lines), 4)
        self.assertIn("Coffee (regular), milk, sugar: £3.05", out.getvalue())


class Structure(unittest.TestCase):
    """The four roles of the pattern."""

    def test_component_and_decorator_are_abstract(self):
        with self.assertRaises(TypeError):
            Beverage()
        with self.assertRaises(TypeError):
            Topping(Coffee())

    def test_every_topping_is_a_beverage(self):
        for cls in TOPPINGS.values():
            self.assertTrue(issubclass(cls, Topping))
            self.assertTrue(issubclass(cls, Beverage))

    def test_topping_refuses_to_wrap_something_that_is_not_a_drink(self):
        with self.assertRaises(TypeError):
            Milk("a coffee, honestly")

    def test_slots_stop_new_attributes_being_attached_by_accident(self):
        with self.assertRaises(AttributeError):
            Milk(Coffee()).colour = "white"


class ToppingContract(unittest.TestCase):
    """Every topping on the menu, over every base, keeps the same contract."""

    def test_each_topping_adds_its_price_and_its_name(self):
        for make_base, topping in itertools.product(BASES.values(), TOPPINGS.values()):
            with self.subTest(base=make_base().description(), topping=topping.name):
                base = make_base()
                drink = topping(base)
                self.assertEqual(drink.cost(), base.cost() + topping.price)
                self.assertEqual(drink.description(), f"{base.description()}, {topping.name}")
                self.assertIs(drink.wrapped, base)

    def test_additive_toppings_give_the_same_price_in_any_order(self):
        toppings = [Milk, Sugar, VanillaSyrup, ExtraShot]
        prices = set()
        for order in itertools.permutations(toppings):
            drink = Espresso()
            for topping in order:
                drink = topping(drink)
            prices.add(drink.cost())
        self.assertEqual(prices, {210 + 40 + 15 + 45 + 60})

    def test_toppings_stack_to_any_depth(self):
        drink = Coffee()
        for _ in range(500):
            drink = Sugar(drink)
        self.assertEqual(drink.cost(), 250 + 500 * 15)
        self.assertEqual(len(drink.layers()), 501)

    def test_prices_are_whole_pence(self):
        drink = make_drink("coffee", ["milk", "sugar", "vanilla syrup"], loyalty=True)
        self.assertIsInstance(drink.cost(), int)
        self.assertEqual(format_price(drink.cost()), "£3.15")

    def test_client_code_treats_plain_and_decorated_drinks_alike(self):
        def till(drinks):
            return sum(drink.cost() for drink in drinks)

        self.assertEqual(till([Coffee(), WhippedCream(OatMilk(Coffee()))]), 250 + 355)


class OrderDesk(unittest.TestCase):
    """make_drink() turns names from an order into decorators safely."""

    def test_builds_the_drink_named_on_the_order(self):
        drink = make_drink("Large Coffee ", ["Oat Milk", "sugar"])
        self.assertEqual(drink.receipt_line(), "Coffee (large), oat milk, sugar: £3.50")

    def test_unknown_names_are_refused(self):
        for base, toppings in [("tea", []), ("coffee", ["ketchup"])]:
            with self.subTest(base=base, toppings=toppings), self.assertRaises(OrderError):
                make_drink(base, toppings)

    def test_names_cannot_reach_attributes_or_other_classes(self):
        hostile = ["__class__", "LoyaltyDiscount", "loyalty discount", "Topping", "__init__"]
        for name in hostile:
            with self.subTest(name=name), self.assertRaises(OrderError):
                make_drink("coffee", [name])

    def test_shop_limits_are_enforced(self):
        with self.assertRaises(OrderError):
            make_drink("coffee", ["sugar"] * (MAX_EACH + 1))
        with self.assertRaises(OrderError):
            make_drink("coffee", ["milk", "sugar", "vanilla syrup", "extra shot", "oat milk", "whipped cream"][: MAX_TOPPINGS + 1])
        make_drink("coffee", ["sugar"] * MAX_EACH)  # the limit itself is allowed

    def test_menu_tables_are_read_only(self):
        with self.assertRaises(TypeError):
            TOPPINGS["free shot"] = ExtraShot
        with self.assertRaises(TypeError):
            BASES["tea"] = Coffee

    def test_a_new_topping_can_be_offered_without_editing_coffee_py(self):
        class Cinnamon(Topping):
            name, price = "cinnamon", 20
            __slots__ = ()

        menu = dict(TOPPINGS, cinnamon=Cinnamon)
        drink = make_drink("espresso", ["cinnamon"], menu=menu)
        self.assertEqual(drink.receipt_line(), "Espresso, cinnamon: £2.30")
        with self.assertRaises(OrderError):
            make_drink("espresso", ["cinnamon"])  # the shop's own menu is unchanged

    def test_function_decorator_audits_orders_and_keeps_the_name(self):
        log = []
        desk = audited(log)(make_drink)
        desk("coffee", ["milk"])
        with self.assertRaises(OrderError):
            desk("coffee", ["ketchup"])
        self.assertEqual(desk.__name__, "make_drink")
        self.assertEqual(len(log), 2)
        self.assertTrue(log[0].startswith("make_drink -> <Milk: Coffee (regular), milk: £2.90>"))
        self.assertTrue(log[1].startswith("make_drink refused:"))


class Limits(unittest.TestCase):
    """Behaviour the pattern does not protect against, recorded on purpose."""

    def test_decorated_drink_is_no_longer_an_instance_of_its_base(self):
        drink = Milk(Espresso())
        self.assertIsInstance(drink, Beverage)
        self.assertNotIsInstance(drink, Espresso)

    def test_members_added_by_a_decorator_are_hidden_by_the_next_layer(self):
        class Labelled(Topping):
            """Adds a new member, as Francesco's InstrumentedHttpClient does."""
            name, price = "name on cup", 0

            def __init__(self, drink, label):
                super().__init__(drink)
                self.label = label

        inner = Labelled(Coffee(), "Ian")
        self.assertEqual(inner.label, "Ian")
        with self.assertRaises(AttributeError):
            Sugar(inner).label

    def test_a_decorator_that_alters_the_price_makes_order_matter(self):
        discount_last = LoyaltyDiscount(VanillaSyrup(Coffee()))
        discount_first = VanillaSyrup(LoyaltyDiscount(Coffee()))
        self.assertEqual(discount_last.cost(), 295 - 29)
        self.assertEqual(discount_first.cost(), 250 - 25 + 45)
        self.assertNotEqual(discount_last.cost(), discount_first.cost())

    def test_order_desk_puts_the_discount_outermost(self):
        drink = make_drink("coffee", ["vanilla syrup"], loyalty=True)
        self.assertEqual(drink.layers()[-1], "loyalty discount")
        self.assertEqual(drink.cost(), 266)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(coffee))
    return tests


if __name__ == "__main__":
    unittest.main()
