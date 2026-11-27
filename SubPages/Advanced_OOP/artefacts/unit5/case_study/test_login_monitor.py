"""Tests for the Observer case study. Run from this folder with:

    python -m unittest -v test_login_monitor
"""

import doctest
import unittest

import login_monitor
from login_monitor import (
    AuditLog,
    LockedAccountAlert,
    LockoutPolicy,
    LoginMonitor,
    MonitoringError,
    Observer,
)


class Broken(Observer):
    def notify(self, event, monitor):
        raise RuntimeError("disk full")


class LeavesAfterFirstEvent(Observer):
    def __init__(self):
        self.seen = 0

    def notify(self, event, monitor):
        self.seen += 1
        monitor.unsubscribe(self)


class ObserverBehaviour(unittest.TestCase):

    def setUp(self):
        self.monitor = LoginMonitor()
        self.audit = AuditLog()
        self.policy = LockoutPolicy(threshold=3)
        self.alert = LockedAccountAlert(self.policy)
        self.monitor.subscribe(self.audit, required=True)
        self.monitor.subscribe(self.policy)
        self.monitor.subscribe(self.alert)

    def fail_login(self, times, user="alice"):
        for _ in range(times):
            self.monitor.record(user, success=False, source="203.0.113.7")

    def test_every_observer_hears_every_event(self):
        self.fail_login(2)
        self.assertEqual(len(self.audit.events), 2)
        self.assertFalse(self.policy.is_locked("alice"))

    def test_lockout_then_alert(self):
        self.fail_login(3)
        self.assertTrue(self.policy.is_locked("alice"))
        self.assertEqual(len(self.alert.alerts), 1)
        self.fail_login(1)
        self.assertEqual(len(self.alert.alerts), 2)

    def test_success_resets_the_count(self):
        self.fail_login(2)
        self.monitor.record("alice", success=True, source="198.51.100.1")
        self.fail_login(2)
        self.assertFalse(self.policy.is_locked("alice"))

    def test_new_response_needs_no_edit_to_the_monitor(self):
        seen = []

        class Counter(Observer):
            def notify(self, event, monitor):
                seen.append(event.user)

        self.monitor.subscribe(Counter())
        self.fail_login(1, user="bob")
        self.assertEqual(seen, ["bob"])

    def test_order_of_subscription_matters(self):
        # The alert reads the policy's state, so it must be notified after it.
        monitor, policy = LoginMonitor(), LockoutPolicy(threshold=1)
        alert = LockedAccountAlert(policy)
        monitor.subscribe(alert)
        monitor.subscribe(policy)
        monitor.record("alice", success=False, source="x")
        self.assertTrue(policy.is_locked("alice"))
        self.assertEqual(alert.alerts, [])  # missed: it ran before the lock


class DesignDecisions(unittest.TestCase):

    def test_required_observer_cannot_be_unsubscribed(self):
        monitor, audit = LoginMonitor(), AuditLog()
        monitor.subscribe(audit, required=True)
        with self.assertRaises(MonitoringError):
            monitor.unsubscribe(audit)
        self.assertIn(audit, monitor.observers)

    def test_optional_observer_can_leave_during_notification(self):
        monitor, leaver, audit = LoginMonitor(), LeavesAfterFirstEvent(), AuditLog()
        monitor.subscribe(leaver)
        monitor.subscribe(audit)
        monitor.record("alice", False, "x")
        monitor.record("alice", False, "x")
        self.assertEqual(leaver.seen, 1)
        self.assertEqual(len(audit.events), 2)  # the copy kept iteration safe

    def test_broken_observer_does_not_silence_the_others(self):
        monitor, audit = LoginMonitor(), AuditLog()
        monitor.subscribe(Broken())
        monitor.subscribe(audit)
        monitor.record("alice", False, "x")
        self.assertEqual(len(audit.events), 1)
        self.assertEqual(monitor.errors, ["Broken: disk full"])

    def test_duplicate_subscription_is_ignored(self):
        monitor, audit = LoginMonitor(), AuditLog()
        monitor.subscribe(audit)
        monitor.subscribe(audit)
        monitor.record("alice", True, "x")
        self.assertEqual(len(audit.events), 1)

    def test_events_cannot_be_altered_by_an_observer(self):
        monitor = LoginMonitor()
        event = monitor.record("alice", False, "x")
        with self.assertRaises(AttributeError):
            event.success = True

    def test_only_observers_can_subscribe(self):
        with self.assertRaises(TypeError):
            LoginMonitor().subscribe(print)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(login_monitor))
    return tests


if __name__ == "__main__":
    unittest.main()
