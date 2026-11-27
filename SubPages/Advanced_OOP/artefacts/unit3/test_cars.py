"""Tests for the Unit 3 Factory Method exercise. Run from this folder with:

    python -m unittest -v test_cars

The first group checks each requirement in the brief. The rest check what
the pattern is meant to give: client code that names no concrete car, a
contract every factory keeps, new car types without edits, and safe serial
numbers across threads. They also record two limits: Python cannot force
callers through a factory, and adding an abstract method to Car breaks every
existing subclass. The doctests in cars.py are collected too (see load_tests
at the end).
"""

import abc
import ast
import contextlib
import doctest
import inspect
import io
import textwrap
import threading
import unittest

import cars
from cars import (
    FACTORIES,
    SUV,
    BuildRecord,
    Car,
    CarFactory,
    Hatchback,
    HatchbackFactory,
    Sedan,
    SedanFactory,
    SerialNumbers,
    SUVFactory,
    factory_for,
    road_test,
    run_production,
)

PAIRS = [(SedanFactory, Sedan), (SUVFactory, SUV), (HatchbackFactory, Hatchback)]
CONCRETE_NAMES = {"Sedan", "SUV", "Hatchback", "SedanFactory", "SUVFactory", "HatchbackFactory"}


def every_factory(serials=None):
    """One instance of every concrete CarFactory in the module, so the
    contract tests cover a new factory as soon as it is registered."""
    serials = serials or SerialNumbers()
    return [maker(serials) for maker in FACTORIES.values()]


def names_used(function):
    """Every bare name that a function's source refers to."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


class ElectricCar(Car):
    """A new product, defined outside cars.py."""
    body_type = "Electric"
    seats = 4

    def drive(self):
        return "Electric: silent acceleration away from the lights"


class ElectricCarFactory(CarFactory):
    def create_car(self):
        return ElectricCar()


class BriefRequirements(unittest.TestCase):
    """The five parts of the brief."""

    def test_car_is_abstract_with_abstract_drive(self):
        self.assertTrue(inspect.isabstract(Car))
        self.assertIn("drive", Car.__abstractmethods__)
        with self.assertRaises(TypeError):
            Car()

    def test_concrete_cars_implement_drive_differently(self):
        sentences = set()
        for cls in (Sedan, SUV, Hatchback):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, Car))
                self.assertIn("drive", cls.__dict__)  # overridden, not inherited
                sentence = cls().drive()
                self.assertIsInstance(sentence, str)
                self.assertTrue(sentence.startswith(cls.body_type))
                sentences.add(sentence)
        self.assertEqual(len(sentences), 3)

    def test_car_factory_is_abstract_with_abstract_create_car(self):
        self.assertTrue(inspect.isabstract(CarFactory))
        self.assertEqual(CarFactory.__abstractmethods__, frozenset({"create_car"}))
        with self.assertRaises(TypeError):
            CarFactory()

    def test_each_concrete_factory_overrides_create_car(self):
        for factory_cls, car_cls in PAIRS:
            with self.subTest(factory=factory_cls.__name__):
                self.assertIn("create_car", factory_cls.__dict__)
                self.assertIs(type(factory_cls().create_car()), car_cls)

    def test_demonstration_builds_and_drives_each_type(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cars.main()
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual([line.split()[1] for line in lines], ["Sedan:", "SUV:", "Hatchback:"])
        self.assertEqual([line.split()[0] for line in lines],
                         ["CAR-000001", "CAR-000002", "CAR-000003"])


class ClientNamesNoConcreteClass(unittest.TestCase):
    """The client code depends only on Car and CarFactory."""

    def test_client_functions_name_no_concrete_class(self):
        for function in (road_test, run_production, CarFactory.produce):
            with self.subTest(function=function.__qualname__):
                self.assertFalse(names_used(function) & CONCRETE_NAMES)

    def test_client_annotations_are_the_abstractions(self):
        hints = inspect.get_annotations(road_test, eval_str=True)
        self.assertIs(hints["factory"], CarFactory)

    def test_main_names_no_concrete_class_either(self):
        # main() asks factory_for(); only the lookup table names the classes.
        self.assertFalse(names_used(cars.main) & CONCRETE_NAMES)


class FactoryContract(unittest.TestCase):
    """Rules every concrete factory must keep, run against all of them."""

    def test_create_car_returns_a_car(self):
        for factory in every_factory():
            with self.subTest(factory=type(factory).__name__):
                self.assertIsInstance(factory.create_car(), Car)

    def test_create_car_returns_a_new_object_each_time(self):
        # A factory that quietly handed back one shared car would be a
        # singleton in disguise.
        for factory in every_factory():
            with self.subTest(factory=type(factory).__name__):
                self.assertIsNot(factory.create_car(), factory.create_car())

    def test_produce_returns_a_build_record(self):
        for factory in every_factory():
            with self.subTest(factory=type(factory).__name__):
                record = factory.produce()
                self.assertIsInstance(record, BuildRecord)
                self.assertEqual(record.factory, type(factory).__name__)
                self.assertTrue(record.car.drive())

    def test_shared_serial_source_gives_unique_increasing_serials(self):
        serials = SerialNumbers()
        records = [factory.produce() for factory in every_factory(serials) * 3]
        numbers = [record.serial for record in records]
        self.assertEqual(numbers, sorted(numbers))
        self.assertEqual(len(set(numbers)), 9)

    def test_factories_with_their_own_source_do_not_share_numbers(self):
        first = SedanFactory().produce().serial
        second = SedanFactory().produce().serial
        self.assertEqual(first, second)  # each starts at CAR-000001


class OpenForNewCarTypes(unittest.TestCase):
    """A new car type is a new pair of classes. Nothing in cars.py changes."""

    def test_new_factory_works_with_existing_client_code(self):
        lines = run_production([ElectricCarFactory(), SedanFactory()])
        self.assertTrue(lines[0].endswith("Electric: silent acceleration away from the lights"))

    def test_new_type_can_be_registered_without_editing_the_module(self):
        registry = dict(FACTORIES, electric=ElectricCarFactory)
        factory = factory_for("electric", SerialNumbers(), registry)
        self.assertIsInstance(factory.produce().car, ElectricCar)

    def test_default_registry_cannot_be_changed_by_other_code(self):
        with self.assertRaises(TypeError):
            FACTORIES["electric"] = ElectricCarFactory  # type: ignore[index]
        self.assertNotIn("electric", FACTORIES)


class Guards(unittest.TestCase):
    """What produce() and factory_for() refuse."""

    def test_produce_rejects_a_factory_that_returns_something_else(self):
        class BrokenFactory(CarFactory):
            def create_car(self):
                return "a car, honestly"

        with self.assertRaises(TypeError) as caught:
            BrokenFactory().produce()
        self.assertIn("BrokenFactory.create_car() returned str", str(caught.exception))

    def test_factory_for_accepts_names_from_a_file(self):
        self.assertIsInstance(factory_for("  SUV \n", SerialNumbers()), SUVFactory)

    def test_factory_for_rejects_unknown_names_and_lists_the_options(self):
        with self.assertRaises(ValueError) as caught:
            factory_for("van", SerialNumbers())
        self.assertIn("hatchback, sedan, suv", str(caught.exception))


class LimitsOfThePattern(unittest.TestCase):
    """Things the pattern does not do in Python. These pass because the
    limits are real."""

    def test_nothing_stops_a_caller_bypassing_the_factory(self):
        # Python has no private constructors, so a Sedan made directly skips
        # produce() and gets no serial number or check.
        car = Sedan()
        self.assertFalse(hasattr(car, "serial"))

    def test_new_abstract_method_on_the_product_breaks_existing_subclasses(self):
        # Reddy (2024, ch. 10): adding a pure virtual method to a published
        # base class breaks every client subclass. In Python the break shows
        # up only when the subclass is instantiated.
        class CarV2(Car):
            @abc.abstractmethod
            def refuel(self):
                """New in version 2."""

        class OldSedan(CarV2):
            body_type, seats = "Sedan", 5

            def drive(self):
                return "Sedan: written against version 1"

        with self.assertRaises(TypeError):
            OldSedan()

    def test_a_default_implementation_keeps_existing_subclasses_working(self):
        class CarV2(Car):
            def refuel(self):
                return "refuelled"

        class OldSedan(CarV2):
            body_type, seats = "Sedan", 5

            def drive(self):
                return "Sedan: written against version 1"

        self.assertEqual(OldSedan().refuel(), "refuelled")


class SerialNumbersAcrossThreads(unittest.TestCase):
    """Reddy (2024, ch. 9): shared state needs synchronised access."""

    def test_parallel_factories_never_repeat_a_serial(self):
        serials = SerialNumbers()
        factories = every_factory(serials) * 3
        results, lock = [], threading.Lock()

        def worker(factory):
            made = [factory.produce().serial for _ in range(300)]
            with lock:
                results.extend(made)

        threads = [threading.Thread(target=worker, args=(f,)) for f in factories]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 2700)
        self.assertEqual(len(set(results)), 2700)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(cars))
    return tests


if __name__ == "__main__":
    unittest.main()
