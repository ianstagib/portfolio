"""Tests for the Unit 5 Strategy exercise. Run from this folder with:

    python -m unittest -v test_password_hashing

The first group checks what the exercise asks of the pattern: algorithms that
can be swapped, chosen while the program runs, behind a context that never
names a concrete strategy. The contract group runs every hashing strategy
through the same checks. The migration group follows a store from unsalted
SHA-256 to PBKDF2 and scrypt. The security group treats stored hashes and
user names as untrusted input. The last group records limits discussed in
the reflection. Cost settings are lowered so the suite runs in seconds; the
doctests in password_hashing.py are collected too (see load_tests).
"""

import ast
import contextlib
import doctest
import hashlib
import inspect
import io
import textwrap
import time
import unittest

import password_hashing
from password_hashing import (
    HashFormatError,
    HashingStrategy,
    LegacySha256,
    PasswordStore,
    Pbkdf2Sha256,
    Scrypt,
    main,
)

FAST_PBKDF2 = Pbkdf2Sha256(iterations=1_000)
FAST_SCRYPT = Scrypt(n=2**10)
LEGACY_HASH = "sha256$" + hashlib.sha256(b"letmein-2019").hexdigest()


def fast_strategies():
    return [Pbkdf2Sha256(iterations=1_000), Scrypt(n=2**10)]


class Reversed(HashingStrategy):
    """A deliberately useless strategy, defined only in this test file."""

    identifier = "reversed"

    def hash(self, password):
        return f"reversed${password[::-1]}"

    def verify(self, password, encoded):
        return encoded == self.hash(password)


class Spy(HashingStrategy):
    """Wraps a real strategy and records every call made to it."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.identifier = inner.identifier

    def hash(self, password):
        self.calls.append("hash")
        return self.inner.hash(password)

    def verify(self, password, encoded):
        self.calls.append("verify")
        return self.inner.verify(password, encoded)


class BriefRequirements(unittest.TestCase):
    """A family of algorithms, each encapsulated and interchangeable."""

    def test_every_strategy_works_behind_the_same_context(self):
        for strategy in fast_strategies():
            with self.subTest(strategy=repr(strategy)):
                store = PasswordStore(strategy)
                store.set_password("alice", "s3cret passphrase")
                self.assertTrue(store.check_password("alice", "s3cret passphrase"))
                self.assertFalse(store.check_password("alice", "wrong"))
                self.assertEqual(store.strategy_of("alice"), strategy.identifier)

    def test_strategy_can_be_changed_while_running(self):
        store = PasswordStore(FAST_PBKDF2)
        store.set_password("alice", "first")
        store.set_strategy(FAST_SCRYPT)
        store.set_password("bob", "second")
        self.assertIs(store.strategy, FAST_SCRYPT)
        self.assertEqual(store.strategy_of("alice"), "pbkdf2_sha256")
        self.assertEqual(store.strategy_of("bob"), "scrypt")
        self.assertTrue(store.check_password("bob", "second"))

    def test_context_names_no_concrete_strategy(self):
        source = textwrap.dedent(inspect.getsource(PasswordStore))
        names = {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)}
        for concrete in ("Pbkdf2Sha256", "Scrypt", "LegacySha256", "hashlib"):
            self.assertNotIn(concrete, names)

    def test_context_has_no_branch_on_algorithm_name(self):
        source = textwrap.dedent(inspect.getsource(PasswordStore))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Compare):
                constants = [c.value for c in ast.walk(node) if isinstance(c, ast.Constant)]
                self.assertFalse({"pbkdf2_sha256", "scrypt", "sha256"} & set(constants))

    def test_strategy_abstraction_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            HashingStrategy()


class StrategyContract(unittest.TestCase):
    """Every concrete strategy keeps the same promises."""

    def test_round_trip_and_wrong_password(self):
        for strategy in fast_strategies():
            with self.subTest(strategy=repr(strategy)):
                encoded = strategy.hash("pässwörd ☕")
                self.assertTrue(strategy.verify("pässwörd ☕", encoded))
                self.assertFalse(strategy.verify("passwörd ☕", encoded))
                self.assertTrue(strategy.owns(encoded))

    def test_same_password_gets_a_fresh_salt_each_time(self):
        for strategy in fast_strategies():
            with self.subTest(strategy=repr(strategy)):
                self.assertNotEqual(strategy.hash("same"), strategy.hash("same"))

    def test_hash_names_its_strategy_and_settings(self):
        self.assertEqual(FAST_PBKDF2.hash("x").split("$")[:2], ["pbkdf2_sha256", "1000"])
        self.assertEqual(FAST_SCRYPT.hash("x").split("$")[:4], ["scrypt", "1024", "8", "1"])

    def test_defaults_follow_owasp_guidance(self):
        self.assertEqual(Pbkdf2Sha256().iterations, 600_000)
        self.assertEqual((Scrypt().n, Scrypt().r, Scrypt().p), (2**17, 8, 1))

    def test_hash_made_with_current_settings_needs_no_rehash(self):
        for strategy in fast_strategies():
            with self.subTest(strategy=repr(strategy)):
                self.assertFalse(strategy.needs_rehash(strategy.hash("x")))

    def test_strategy_refuses_another_strategys_hash(self):
        with self.assertRaises(HashFormatError):
            FAST_SCRYPT.verify("x", FAST_PBKDF2.hash("x"))

    def test_invalid_settings_are_refused_at_construction(self):
        for make in (lambda: Pbkdf2Sha256(iterations=10), lambda: Scrypt(n=1000),
                     lambda: Scrypt(n=2**21), lambda: Scrypt(r=0)):
            with self.assertRaises(ValueError):
                make()


class OpenForExtension(unittest.TestCase):
    """A new strategy needs no edit to password_hashing.py."""

    def test_strategy_defined_elsewhere_plugs_in(self):
        store = PasswordStore(Reversed())
        store.set_password("alice", "abc")
        self.assertEqual(store.records["alice"], "reversed$cba")
        self.assertTrue(store.check_password("alice", "abc"))

    def test_switching_to_a_new_strategy_upgrades_old_hashes(self):
        store = PasswordStore(Reversed())
        store.set_password("alice", "abc")
        store.set_strategy(FAST_PBKDF2)
        self.assertTrue(store.check_password("alice", "abc"))
        self.assertEqual(store.strategy_of("alice"), "pbkdf2_sha256")


class Migration(unittest.TestCase):
    """Old hashes keep working and are replaced at the next login."""

    def make_store(self):
        store = PasswordStore(FAST_PBKDF2, accepted=[LegacySha256()])
        store.import_hash("alice", LEGACY_HASH)
        return store

    def test_legacy_hash_verifies_and_is_upgraded(self):
        store = self.make_store()
        self.assertTrue(store.check_password("alice", "letmein-2019"))
        self.assertEqual(store.strategy_of("alice"), "pbkdf2_sha256")
        self.assertTrue(store.check_password("alice", "letmein-2019"))

    def test_failed_login_does_not_upgrade(self):
        store = self.make_store()
        self.assertFalse(store.check_password("alice", "guess"))
        self.assertEqual(store.records["alice"], LEGACY_HASH)

    def test_weaker_settings_of_the_same_algorithm_are_upgraded(self):
        old = Pbkdf2Sha256(iterations=1_000)
        store = PasswordStore(old)
        store.set_password("alice", "pw")
        store.set_strategy(Pbkdf2Sha256(iterations=2_000))
        store.check_password("alice", "pw")
        self.assertEqual(store.records["alice"].split("$")[1], "2000")

    def test_previous_strategy_stays_accepted_after_a_switch(self):
        store = PasswordStore(FAST_PBKDF2)
        store.set_password("alice", "pw")
        store.set_strategy(FAST_SCRYPT)
        store.set_strategy(Reversed())
        self.assertTrue(store.check_password("alice", "pw"))

    def test_demonstration_shows_the_migration(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            lines = main()
        self.assertEqual(len(lines), 4)
        self.assertIn("alice=sha256", lines[0])
        self.assertIn("alice=scrypt, bob=scrypt", lines[3])
        self.assertIn("alice logs in", out.getvalue())


class StoredDataIsUntrusted(unittest.TestCase):
    """A stored hash or a user name may have been chosen by an attacker."""

    def test_unknown_strategy_is_refused(self):
        store = PasswordStore(FAST_PBKDF2)
        with self.assertRaises(HashFormatError):
            store.import_hash("mallory", "md5$5f4dcc3b5aa765d61d8327deb882cf99")

    def test_legacy_hashes_are_refused_unless_explicitly_accepted(self):
        with self.assertRaises(HashFormatError):
            PasswordStore(FAST_PBKDF2).import_hash("alice", LEGACY_HASH)

    def test_tampered_cost_is_refused_quickly(self):
        good = FAST_PBKDF2.hash("pw").split("$")
        tampered = "$".join([good[0], "9" * 12, good[2], good[3]])
        start = time.perf_counter()
        with self.assertRaises(HashFormatError):
            FAST_PBKDF2.verify("pw", tampered)
        self.assertLess(time.perf_counter() - start, 0.5)

    def test_tampered_scrypt_memory_is_refused(self):
        good = FAST_SCRYPT.hash("pw").split("$")
        for n in (str(2**30), "1000", "-1"):
            with self.subTest(n=n), self.assertRaises(HashFormatError):
                FAST_SCRYPT.verify("pw", "$".join([good[0], n, *good[2:]]))

    def test_malformed_hashes_are_refused(self):
        for bad in ("pbkdf2_sha256$1000$salt", "pbkdf2_sha256$1000$!!!$AAAA",
                    "pbkdf2_sha256$1e3$AAAA$AAAA", ""):
            with self.subTest(bad=bad), self.assertRaises(HashFormatError):
                FAST_PBKDF2.verify("pw", bad)

    def test_unknown_user_still_costs_one_hash(self):
        spy = Spy(FAST_PBKDF2)
        store = PasswordStore(spy)
        spy.calls.clear()
        self.assertFalse(store.check_password("nobody", "pw"))
        self.assertIn("verify", spy.calls)

    def test_records_are_read_only(self):
        store = PasswordStore(FAST_PBKDF2)
        store.set_password("alice", "pw")
        with self.assertRaises(TypeError):
            store.records["alice"] = "reversed$wp"

    def test_empty_or_non_string_password_is_refused(self):
        store = PasswordStore(FAST_PBKDF2)
        for bad in ("", None, b"bytes"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                store.set_password("alice", bad)

    def test_digest_comparison_is_constant_time(self):
        for cls in (Pbkdf2Sha256, Scrypt, LegacySha256):
            self.assertIn("hmac.compare_digest", inspect.getsource(cls.verify))


class RecordedLimits(unittest.TestCase):
    """Behaviour the reflection discusses rather than hides."""

    def test_choosing_a_strategy_still_needs_a_lookup(self):
        # The if/elif chain is gone, but the store still maps a name in the
        # stored hash to a strategy object: the decision has moved, not vanished.
        source = inspect.getsource(PasswordStore._strategy_for)
        self.assertIn("self._accepted.get(identifier)", source)

    def test_legacy_strategy_cannot_stand_in_for_the_others(self):
        # LegacySha256 is a HashingStrategy that refuses hash(), so it is not
        # fully substitutable (Liskov). The context accepts it all the same.
        store = PasswordStore(LegacySha256())
        with self.assertRaises(NotImplementedError):
            store.set_password("alice", "pw")

    def test_a_plain_function_is_not_accepted_as_a_strategy(self):
        # Nesteruk (2022, ch. 22) shows strategies passed as lambdas. This store
        # needs an identifier and verify() as well, so it insists on a class.
        with self.assertRaises(TypeError):
            PasswordStore(FAST_PBKDF2).set_strategy(lambda pw: pw[::-1])


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(password_hashing))
    return tests


if __name__ == "__main__":
    unittest.main()
