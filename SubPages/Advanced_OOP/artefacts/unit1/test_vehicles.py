"""Tests for vehicles.py. Run from this folder with:

    python -m unittest -v test_vehicles

The doctests in vehicles.py are collected too (see load_tests at the end).
"""

import doctest
import gc
import platform
import unittest

import vehicles
from vehicles import Bike, Car, Driver, Fleet, Truck, Vehicle

VIN = "WF0XXXGCDX1234567"


def make_car(registration="AB12CDE", **kwargs):
    return Car(registration, "Ford", "Focus", 2021, vin=VIN, **kwargs)


def make_bike(registration="BK19ABC", engine_cc=125, power_kw=11):
    return Bike(registration, "Honda", "CB125R", 2022, vin=VIN,
                engine_cc=engine_cc, power_kw=power_kw)


def make_truck(registration="TR70XYZ", gross_weight_kg=18000, kerb_weight_kg=8000):
    return Truck(registration, "DAF", "LF", 2020, vin=VIN,
                 gross_weight_kg=gross_weight_kg, kerb_weight_kg=kerb_weight_kg)


class AbstractionTests(unittest.TestCase):

    def test_vehicle_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            Vehicle("AB12CDE", "Ford", "Focus", 2021, vin=VIN)

    def test_abstract_interface(self):
        self.assertEqual(Vehicle.__abstractmethods__, {"wheels", "licence_category"})

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class Tram(Vehicle):
            @property
            def wheels(self):
                return 8
            # licence_category is missing

        with self.assertRaises(TypeError):
            Tram("TM01ABC", "Siemens", "Avenio", 2015, vin=VIN)


class InheritanceTests(unittest.TestCase):

    def test_subclasses_are_vehicles(self):
        for vehicle in (make_car(), make_bike(), make_truck()):
            with self.subTest(type(vehicle).__name__):
                self.assertIsInstance(vehicle, Vehicle)

    def test_method_resolution_order(self):
        self.assertEqual(Truck.__mro__[:2], (Truck, Vehicle))

    def test_base_constructor_validation_is_inherited(self):
        builders = {
            "Car": lambda reg: make_car(reg),
            "Bike": lambda reg: make_bike(reg),
            "Truck": lambda reg: make_truck(reg),
        }
        for name, build in builders.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    build("NOT A REG!")

    def test_registration_is_normalised(self):
        self.assertEqual(make_car("ab12 cde").registration, "AB12CDE")

    def test_year_is_checked(self):
        with self.assertRaises(ValueError):
            Car("AB12CDE", "Ford", "Model T", 1800, vin=VIN)
        with self.assertRaises(TypeError):
            Car("AB12CDE", "Ford", "Focus", "2021", vin=VIN)


class PolymorphismTests(unittest.TestCase):

    def test_each_subclass_answers_in_its_own_way(self):
        expected = [(make_car(), 4, "B"), (make_bike(), 2, "A1"), (make_truck(), 6, "C")]
        for vehicle, wheels, category in expected:
            with self.subTest(type(vehicle).__name__):
                self.assertEqual(vehicle.wheels, wheels)
                self.assertEqual(vehicle.licence_category, category)

    def test_describe_uses_subclass_hook(self):
        self.assertIn("4 passenger seats", make_car().describe())
        self.assertIn("125 cc", make_bike().describe())
        self.assertIn("0 of 10,000 kg loaded", make_truck().describe())

    def test_bike_licence_boundaries(self):
        cases = [
            (125, 11, "A1"),    # the A1 limits exactly
            (125, 11.1, "A2"),  # just over the A1 power limit
            (126, 9, "A2"),     # just over the A1 engine limit
            (650, 35, "A2"),    # the A2 power limit exactly
            (650, 35.1, "A"),
            (0, 10, "A1"),      # electric
        ]
        for cc, kw, category in cases:
            with self.subTest(cc=cc, kw=kw):
                self.assertEqual(make_bike(engine_cc=cc, power_kw=kw).licence_category, category)

    def test_truck_licence_boundaries(self):
        self.assertEqual(make_truck(gross_weight_kg=7500, kerb_weight_kg=4000).licence_category, "C1")
        self.assertEqual(make_truck(gross_weight_kg=7501, kerb_weight_kg=4000).licence_category, "C")

    def test_fleet_report_is_polymorphic(self):
        fleet = Fleet("Depot")
        for vehicle in (make_car(), make_bike(), make_truck()):
            fleet.add(vehicle)
        report = fleet.report()
        self.assertEqual(len(report), 3)
        self.assertEqual(len(set(report)), 3)


class EncapsulationTests(unittest.TestCase):

    def test_odometer_cannot_be_set_directly(self):
        car = make_car()
        with self.assertRaises(AttributeError):
            car.odometer_km = 0

    def test_odometer_only_goes_forwards(self):
        car = make_car()
        car.record_trip(100)
        for bad in (0, -50, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    car.record_trip(bad)
        with self.assertRaises(TypeError):
            car.record_trip("10")
        self.assertEqual(car.odometer_km, 100)

    def test_truck_cannot_be_overloaded(self):
        truck = make_truck()
        truck.load(9000)
        with self.assertRaises(ValueError):
            truck.load(1001)
        self.assertEqual(truck.load_kg, 9000)  # a refused load changes nothing
        with self.assertRaises(ValueError):
            truck.unload(9001)
        truck.unload(9000)
        self.assertEqual(truck.load_kg, 0)

    def test_constructor_rejects_impossible_vehicles(self):
        with self.assertRaises(ValueError):
            make_car(passenger_seats=9)
        with self.assertRaises(ValueError):
            make_truck(gross_weight_kg=3500, kerb_weight_kg=2000)
        with self.assertRaises(ValueError):
            make_truck(gross_weight_kg=7500, kerb_weight_kg=7500)


class AccessControlTests(unittest.TestCase):

    def test_private_vin_is_name_mangled(self):
        car = make_car()
        self.assertFalse(hasattr(car, "__vin"))
        # Name mangling is a convention, not a security control:
        self.assertEqual(car._Vehicle__vin, VIN)

    def test_vin_is_masked_and_kept_out_of_repr(self):
        car = make_car()
        self.assertEqual(car.vin_masked, "*************4567")
        self.assertNotIn(VIN, repr(car))
        self.assertNotIn(VIN, car.describe())
        self.assertTrue(car.matches_vin(VIN.lower()))
        self.assertFalse(car.matches_vin("WF0XXXGCDX7654321"))

    def test_invalid_vin_is_not_echoed_in_error(self):
        bad_vin = "IOQ12345678901234"
        with self.assertRaises(ValueError) as context:
            Car("AB12CDE", "Ford", "Focus", 2021, vin=bad_vin)
        self.assertNotIn(bad_vin, str(context.exception))

    def test_driver_must_hold_the_right_category(self):
        fleet = Fleet("Depot")
        fleet.add(make_car())
        fleet.add(make_truck())
        car_driver = Driver("Sam", frozenset({"B"}))
        hgv_driver = Driver("Alex", frozenset({"B", "C"}))
        fleet.assign("AB12CDE", car_driver)
        with self.assertRaises(PermissionError):
            fleet.assign("TR70XYZ", car_driver)
        fleet.assign("TR70XYZ", hgv_driver)
        self.assertEqual(fleet.driver_of("tr70 xyz"), hgv_driver)

    def test_higher_categories_cover_lower_ones(self):
        self.assertTrue(Driver("Kim", frozenset({"A"})).can_drive(make_bike()))
        self.assertTrue(Driver("Kim", frozenset({"C"})).can_drive(
            make_truck(gross_weight_kg=7500, kerb_weight_kg=4000)))
        self.assertFalse(Driver("Kim", frozenset({"C1"})).can_drive(make_truck()))
        self.assertFalse(Driver("Kim", frozenset({"XYZ"})).can_drive(make_car()))

    def test_driver_is_immutable(self):
        driver = Driver("Sam", frozenset({"B"}))
        with self.assertRaises(AttributeError):
            driver.licence_categories = frozenset({"C"})

    def test_fleet_rejects_duplicates_and_non_vehicles(self):
        fleet = Fleet("Depot")
        fleet.add(make_car())
        with self.assertRaises(ValueError):
            fleet.add(make_car("AB12 CDE"))
        with self.assertRaises(TypeError):
            fleet.add("AB12CDE")


@unittest.skipUnless(platform.python_implementation() == "CPython",
                     "finaliser timing relies on CPython reference counting")
class DestructorTests(unittest.TestCase):

    def setUp(self):
        gc.collect()
        Vehicle.decommissioned.clear()

    def test_del_on_the_only_reference_runs_the_finaliser(self):
        car = make_car()
        del car
        self.assertEqual(Vehicle.decommissioned, ["AB12CDE"])

    def test_del_does_not_finalise_an_object_still_referenced(self):
        fleet = Fleet("Depot")
        car = make_car()
        fleet.add(car)
        del car  # removes a name, not the object: the fleet still holds it
        self.assertEqual(Vehicle.decommissioned, [])
        fleet.remove("AB12CDE")  # the returned vehicle is dropped straight away
        self.assertEqual(Vehicle.decommissioned, ["AB12CDE"])

    def test_half_built_object_is_not_recorded(self):
        def try_to_build():
            try:
                make_car(passenger_seats=12)
            except ValueError:
                pass

        try_to_build()
        gc.collect()
        self.assertEqual(Vehicle.decommissioned, [])


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(vehicles))
    return tests


if __name__ == "__main__":
    unittest.main()
