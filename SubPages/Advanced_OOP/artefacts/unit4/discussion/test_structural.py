"""Tests for the Unit 4 collaborative discussion code. Run from this folder with:

    python -m unittest -v test_structural

One class of tests per task. The doctests in the three modules are collected
too (see load_tests).
"""

import doctest
import itertools
import unittest
import xml.etree.ElementTree as ET

import file_system_composite
import payment_adapter
import remote_bridge
from file_system_composite import File, FileSystemComponent, Folder
from payment_adapter import LegacySoapPaymentService, PaymentError, SoapPaymentAdapter, checkout
from remote_bridge import TV, AdvancedRemote, Radio, RemoteControl


def order(**changes):
    base = {"order_id": "A1", "amount": "12.50", "currency": "GBP", "card_token": "tok_ok"}
    return base | changes


class RecordingService(LegacySoapPaymentService):
    """Keeps the last envelope so tests can inspect what the adapter sent."""

    def process_payment(self, envelope):
        self.last = envelope
        return super().process_payment(envelope)


class AdapterTask(unittest.TestCase):
    def setUp(self):
        self.service = RecordingService()
        self.gateway = SoapPaymentAdapter(self.service)

    def test_json_request_becomes_soap_with_amount_in_pence(self):
        checkout(self.gateway, order())
        sent = ET.fromstring(self.service.last)
        self.assertEqual(sent.findtext("Body/ProcessPayment/AmountPence"), "1250")

    def test_soap_status_codes_become_json_statuses(self):
        self.assertEqual(checkout(self.gateway, order())["status"], "approved")
        self.assertEqual(checkout(self.gateway, order(card_token="tok_bad"))["status"], "declined")

    def test_invalid_requests_are_refused_before_the_legacy_call(self):
        for bad in (order(currency="EUR"), order(amount="12.505"), order(amount="lots")):
            with self.subTest(bad=bad), self.assertRaises(PaymentError):
                checkout(self.gateway, bad)
        self.assertFalse(hasattr(self.service, "last"))

    def test_values_cannot_inject_xml(self):
        hostile = "</CardToken><AmountPence>1</AmountPence><CardToken>tok_ok"
        result = checkout(self.gateway, order(card_token=hostile))
        self.assertEqual(result["status"], "declined")
        self.assertEqual(ET.fromstring(self.service.last).findtext("Body/ProcessPayment/CardToken"), hostile)

    def test_soap_fault_becomes_an_exception(self):
        class Broken(LegacySoapPaymentService):
            def process_payment(self, envelope):
                return "<Envelope><Body><Fault><Reason>Down</Reason></Fault></Body></Envelope>"

        with self.assertRaisesRegex(PaymentError, "Down"):
            checkout(SoapPaymentAdapter(Broken()), order())

    def test_neither_side_knows_about_the_other(self):
        import inspect
        self.assertNotIn("Soap", inspect.getsource(checkout))
        self.assertNotIn("json", inspect.getsource(LegacySoapPaymentService).lower())


class BridgeTask(unittest.TestCase):
    def test_every_remote_works_with_every_device(self):
        for remote_cls, device_cls in itertools.product((RemoteControl, AdvancedRemote), (TV, Radio)):
            with self.subTest(remote=remote_cls.__name__, device=device_cls.__name__):
                remote = remote_cls(device_cls())
                remote.toggle_power()
                remote.volume_up()
                remote.channel_up()
                self.assertTrue(remote.device.is_enabled())
                self.assertEqual((remote.device.get_volume(), remote.device.get_channel()), (40, 2))

    def test_advanced_remote_adds_controls(self):
        remote = AdvancedRemote(TV())
        remote.mute()
        remote.set_channel(101)
        self.assertEqual(remote.device.status(), "TV: off, channel 101, volume 0")

    def test_devices_enforce_their_own_limits(self):
        with self.assertRaises(ValueError):
            AdvancedRemote(Radio()).set_channel(21)
        remote = RemoteControl(Radio())
        for _ in range(20):
            remote.volume_up()
        self.assertEqual(remote.device.get_volume(), 100)

    def test_channel_up_wraps_round(self):
        radio = Radio()
        radio.set_channel(20)
        RemoteControl(radio).channel_up()
        self.assertEqual(radio.get_channel(), 1)

    def test_a_new_device_needs_no_new_remote(self):
        class Projector(remote_bridge.Device):
            name, max_channel = "Projector", 4

            def status(self):
                return f"Projector input {self._channel}"

        remote = AdvancedRemote(Projector())
        remote.set_channel(3)
        self.assertEqual(remote.device.status(), "Projector input 3")

    def test_remotes_use_only_the_device_interface(self):
        import inspect
        for cls in (RemoteControl, AdvancedRemote):
            source = inspect.getsource(cls)
            self.assertNotIn("TV", source)
            self.assertNotIn("Radio", source)
            self.assertNotIn("._", source.replace("self.device.", ""))


class CompositeTask(unittest.TestCase):
    def setUp(self):
        self.root = Folder("root")
        self.docs = self.root.add(Folder("docs"))
        self.docs.add(File("a.txt", 3))
        self.docs.add(Folder("empty"))
        self.root.add(File("b.txt", 5))

    def test_files_and_folders_share_one_interface(self):
        for item in (File("x", 1), Folder("y")):
            self.assertIsInstance(item, FileSystemComponent)
            self.assertIsInstance(item.size(), int)
            self.assertIsInstance(item.display(), str)

    def test_folder_size_is_the_sum_of_its_tree(self):
        self.assertEqual(self.root.size(), 8)
        self.assertEqual(self.docs.size(), 3)
        self.assertEqual(Folder("new").size(), 0)

    def test_display_shows_the_tree(self):
        self.assertEqual(
            self.root.display().splitlines(),
            ["root/ (8 KB)", "  docs/ (3 KB)", "    a.txt (3 KB)", "    empty/ (0 KB)", "  b.txt (5 KB)"],
        )

    def test_paths_and_removal(self):
        self.assertEqual(self.docs.path(), "root/docs")
        moved = self.root.remove("b.txt")
        self.assertIsNone(moved.parent)
        self.docs.add(moved)
        self.assertEqual(moved.path(), "root/docs/b.txt")
        self.assertEqual(self.root.size(), 8)

    def test_cycles_duplicates_and_bad_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "inside itself"):
            self.docs.add(self.root)  # root is an ancestor of docs
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.root.add(File("b.txt", 1))
        with self.assertRaisesRegex(ValueError, "already in"):
            self.root.add(next(item for item in self.docs if item.name == "a.txt"))
        for bad in ("", "..", "a/b"):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                File(bad, 1)

    def test_deep_trees_work(self):
        folder = self.root
        for depth in range(200):
            folder = folder.add(Folder(f"d{depth}"))
        folder.add(File("deep.txt", 7))
        self.assertEqual(self.root.size(), 15)


def load_tests(loader, tests, ignore):
    for module in (payment_adapter, remote_bridge, file_system_composite):
        tests.addTests(doctest.DocTestSuite(module))
    return tests


if __name__ == "__main__":
    unittest.main()
