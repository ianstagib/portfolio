"""The Unit 5 practical exercise: password storage built with the Strategy pattern.

The unit asks for the Strategy pattern to be applied in a project, so that an
algorithm can be chosen and replaced without editing the code that uses it.
Password hashing is a security problem where that matters in practice: the
recommended algorithm and its cost settings change over time, and a system
has to keep verifying old hashes while it writes new ones with the current
choice. The pattern's three roles each have a class here:

* Strategy: ``HashingStrategy``, an abstract class with ``hash()``,
  ``verify()`` and ``needs_rehash()``.
* Concrete strategies: ``Pbkdf2Sha256`` and ``Scrypt``, both from the
  standard library's ``hashlib``, and ``LegacySha256``, which can only verify
  the unsalted hashes an older system left behind and refuses to make new
  ones.
* Context: ``PasswordStore``, which hashes every new password with its
  current strategy, verifies each stored hash with the strategy named inside
  it, and upgrades a hash to the current strategy when its owner next logs
  in.

Every stored hash names its strategy and settings, for example
``pbkdf2_sha256$600000$<salt>$<digest>``, so a store can hold hashes from
several strategies at once.

>>> store = PasswordStore(Pbkdf2Sha256(iterations=1_000))
>>> store.set_password("alice", "correct horse battery staple")
>>> store.check_password("alice", "correct horse battery staple")
True
>>> store.check_password("alice", "Tr0ub4dor&3")
False
>>> store.records["alice"].split("$")[:2]
['pbkdf2_sha256', '1000']
"""

from __future__ import annotations

import abc
import base64
import hashlib
import hmac
import secrets
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import ClassVar

__all__ = [
    "HashingStrategy",
    "Pbkdf2Sha256",
    "Scrypt",
    "LegacySha256",
    "PasswordStore",
    "HashFormatError",
    "main",
]


class HashFormatError(ValueError):
    """A stored hash is malformed, unknown or asks for unsafe settings."""


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64decode(text: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise HashFormatError("stored hash is not valid base64") from error


def _parse_int(text: str, name: str, low: int, high: int) -> int:
    """Read a cost setting from a stored hash and refuse values out of range.

    A stored hash is data, and a database row that has been tampered with
    could ask for a billion iterations or gigabytes of memory. Checking the
    range stops verification being turned into a denial of service.
    """
    if not text.isdigit():
        raise HashFormatError(f"{name} is not a whole number")
    value = int(text)
    if not low <= value <= high:
        raise HashFormatError(f"{name} {value} is outside {low}..{high}")
    return value


# --- Strategy ------------------------------------------------------------------


class HashingStrategy(abc.ABC):
    """One way of turning a password into something safe to store."""

    identifier: ClassVar[str]

    @abc.abstractmethod
    def hash(self, password: str) -> str:
        """Return a self-describing hash of ``password`` with a fresh salt."""

    @abc.abstractmethod
    def verify(self, password: str, encoded: str) -> bool:
        """Check ``password`` against a hash this strategy produced."""

    def needs_rehash(self, encoded: str) -> bool:
        """Whether a hash made by this strategy used weaker settings than now."""
        return False

    def owns(self, encoded: str) -> bool:
        """Whether ``encoded`` names this strategy."""
        return encoded.split("$", 1)[0] == self.identifier

    def _fields(self, encoded: str, count: int) -> list[str]:
        fields = encoded.split("$")
        if len(fields) != count or fields[0] != self.identifier:
            raise HashFormatError(f"not a {self.identifier} hash")
        return fields[1:]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# --- Concrete strategies ---------------------------------------------------------


class Pbkdf2Sha256(HashingStrategy):
    """PBKDF2 with HMAC-SHA256, from ``hashlib.pbkdf2_hmac``.

    The default of 600,000 iterations follows the OWASP Password Storage
    Cheat Sheet. Tests pass a lower number so that they run quickly.
    """

    identifier = "pbkdf2_sha256"
    DEFAULT_ITERATIONS: ClassVar[int] = 600_000
    MAX_ITERATIONS: ClassVar[int] = 10_000_000

    def __init__(self, iterations: int = DEFAULT_ITERATIONS, salt_bytes: int = 16) -> None:
        if not 1_000 <= iterations <= self.MAX_ITERATIONS:
            raise ValueError("iterations must be between 1,000 and 10,000,000")
        self.iterations = iterations
        self.salt_bytes = salt_bytes

    def _digest(self, password: str, salt: bytes, iterations: int) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)

    def hash(self, password: str) -> str:
        salt = secrets.token_bytes(self.salt_bytes)
        digest = self._digest(password, salt, self.iterations)
        return f"{self.identifier}${self.iterations}${_b64encode(salt)}${_b64encode(digest)}"

    def verify(self, password: str, encoded: str) -> bool:
        iterations_text, salt_text, digest_text = self._fields(encoded, 4)
        iterations = _parse_int(iterations_text, "iterations", 1_000, self.MAX_ITERATIONS)
        expected = _b64decode(digest_text)
        actual = self._digest(password, _b64decode(salt_text), iterations)
        return hmac.compare_digest(actual, expected)

    def needs_rehash(self, encoded: str) -> bool:
        return int(self._fields(encoded, 4)[0]) < self.iterations

    def __repr__(self) -> str:
        return f"Pbkdf2Sha256(iterations={self.iterations})"


class Scrypt(HashingStrategy):
    """scrypt, a memory-hard function, from ``hashlib.scrypt``.

    The defaults (N = 2**17, r = 8, p = 1) follow the OWASP Password Storage
    Cheat Sheet and need about 128 MiB of memory per hash, so ``maxmem`` is
    raised above OpenSSL's 32 MiB default.
    """

    identifier = "scrypt"
    MAX_N: ClassVar[int] = 2**20

    def __init__(self, n: int = 2**17, r: int = 8, p: int = 1, salt_bytes: int = 16) -> None:
        if n < 2 or n & (n - 1) or n > self.MAX_N:
            raise ValueError("n must be a power of two no larger than 2**20")
        if not (1 <= r <= 32 and 1 <= p <= 16):
            raise ValueError("r must be 1..32 and p must be 1..16")
        self.n, self.r, self.p = n, r, p
        self.salt_bytes = salt_bytes

    @staticmethod
    def _digest(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
        maxmem = 128 * n * r * p + 2**20  # what scrypt needs, plus a margin
        return hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=maxmem, dklen=32
        )

    def hash(self, password: str) -> str:
        salt = secrets.token_bytes(self.salt_bytes)
        digest = self._digest(password, salt, self.n, self.r, self.p)
        return (
            f"{self.identifier}${self.n}${self.r}${self.p}$"
            f"{_b64encode(salt)}${_b64encode(digest)}"
        )

    def verify(self, password: str, encoded: str) -> bool:
        n_text, r_text, p_text, salt_text, digest_text = self._fields(encoded, 6)
        n = _parse_int(n_text, "n", 2, self.MAX_N)
        if n & (n - 1):
            raise HashFormatError("n is not a power of two")
        r = _parse_int(r_text, "r", 1, 32)
        p = _parse_int(p_text, "p", 1, 16)
        expected = _b64decode(digest_text)
        actual = self._digest(password, _b64decode(salt_text), n, r, p)
        return hmac.compare_digest(actual, expected)

    def needs_rehash(self, encoded: str) -> bool:
        n, r, p = (int(x) for x in self._fields(encoded, 6)[:3])
        return n < self.n or r < self.r or p < self.p

    def __repr__(self) -> str:
        return f"Scrypt(n=2**{self.n.bit_length() - 1}, r={self.r}, p={self.p})"


class LegacySha256(HashingStrategy):
    """Unsalted SHA-256, as an older system might have stored it.

    It exists only so that old hashes can be checked once and replaced. It
    will not produce a new hash, and every hash it verifies needs rehashing.
    """

    identifier = "sha256"

    def hash(self, password: str) -> str:
        raise NotImplementedError("unsalted SHA-256 is accepted for migration only")

    def verify(self, password: str, encoded: str) -> bool:
        (hex_digest,) = self._fields(encoded, 2)
        actual = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(actual, hex_digest.lower())

    def needs_rehash(self, encoded: str) -> bool:
        return True


# --- Context -----------------------------------------------------------------------


class PasswordStore:
    """Stores password hashes and checks logins, delegating all hashing.

    ``current`` hashes every new password. ``accepted`` lists any further
    strategies whose old hashes may still be verified. The store never
    names a concrete strategy, so a new one can be used without editing it.
    """

    def __init__(
        self, current: HashingStrategy, accepted: Iterable[HashingStrategy] = ()
    ) -> None:
        self._records: dict[str, str] = {}
        self._dummy: str | None = None
        self._accepted: dict[str, HashingStrategy] = {}
        for strategy in accepted:
            self._accepted[strategy.identifier] = strategy
        self._current = current
        self.set_strategy(current)

    @property
    def strategy(self) -> HashingStrategy:
        return self._current

    @property
    def records(self) -> Mapping[str, str]:
        """A read-only view of the stored hashes, by user name."""
        return MappingProxyType(self._records)

    def set_strategy(self, strategy: HashingStrategy) -> None:
        """Change the algorithm used for new hashes, while the program runs.

        The previous strategy stays accepted, so hashes it made still verify
        and are upgraded at their owners' next login.
        """
        if not isinstance(strategy, HashingStrategy):
            raise TypeError("strategy must be a HashingStrategy")
        self._accepted.setdefault(self._current.identifier, self._current)
        self._accepted[strategy.identifier] = strategy
        self._current = strategy

    def set_password(self, user: str, password: str) -> None:
        if not isinstance(password, str) or not password:
            raise ValueError("password must be a non-empty string")
        self._records[user] = self._current.hash(password)

    def import_hash(self, user: str, encoded: str) -> None:
        """Load a hash from an older system, if an accepted strategy can read it."""
        self._strategy_for(encoded)
        self._records[user] = encoded

    def check_password(self, user: str, password: str) -> bool:
        """Verify a login and upgrade the stored hash if it is out of date.

        An unknown user still costs one hash with the current strategy, so the
        time taken does not reveal which user names exist.
        """
        encoded = self._records.get(user)
        if encoded is None:
            self._current.verify(password, self._dummy_hash())
            return False
        strategy = self._strategy_for(encoded)
        if not strategy.verify(password, encoded):
            return False
        if strategy is not self._current or strategy.needs_rehash(encoded):
            self._records[user] = self._current.hash(password)
        return True

    def strategy_of(self, user: str) -> str:
        return self._records[user].split("$", 1)[0]

    def _strategy_for(self, encoded: str) -> HashingStrategy:
        identifier = encoded.split("$", 1)[0]
        strategy = self._accepted.get(identifier)
        if strategy is None:
            raise HashFormatError(f"no accepted strategy for {identifier!r}")
        return strategy

    def _dummy_hash(self) -> str:
        if self._dummy is None or not self._current.owns(self._dummy):
            self._dummy = self._current.hash(secrets.token_urlsafe(16))
        return self._dummy


# --- Demonstration -----------------------------------------------------------------


def main(fast: bool = True) -> list[str]:
    """Walk through a migration from unsalted SHA-256 to PBKDF2 and then scrypt.

    ``fast`` lowers the cost settings so the demonstration runs in a moment;
    a real system would use the defaults.
    """
    pbkdf2 = Pbkdf2Sha256(iterations=1_000) if fast else Pbkdf2Sha256()
    scrypt = Scrypt(n=2**10) if fast else Scrypt()
    store = PasswordStore(pbkdf2, accepted=[LegacySha256()])
    lines: list[str] = []

    def show(event: str) -> None:
        summary = ", ".join(f"{u}={store.strategy_of(u)}" for u in sorted(store.records))
        lines.append(f"{event:<38} {summary}")

    store.import_hash("alice", "sha256$" + hashlib.sha256(b"letmein-2019").hexdigest())
    show("alice imported from the old system")
    store.check_password("alice", "letmein-2019")
    show("alice logs in")
    store.set_strategy(scrypt)
    store.set_password("bob", "a long passphrase for bob")
    show("strategy switched; bob registers")
    store.check_password("alice", "letmein-2019")
    show("alice logs in again")
    for line in lines:
        print(line)
    return lines


if __name__ == "__main__":
    main()
