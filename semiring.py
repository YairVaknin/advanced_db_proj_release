"""
This module defines various semiring classes used for provenance computations.
Each semiring implements the required operations and provenance token generation.
"""

import hashlib
import random
from abc import ABC, abstractmethod
from typing import Any
from pandas import isna
import sympy  # type: ignore


def generate_deterministic_random(table_name: str, idx: int) -> random.Random:
    """
    Generates a deterministic random.Random instance seeded based on table_name and idx.

    Args:
        table_name (str): The name of the table.
        idx (int): The index of the row.

    Returns:
        random.Random: A random.Random instance seeded based on the input parameters.
    """
    seed_str = f"{table_name}_{idx}"
    seed_bytes = seed_str.encode("utf-8")
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    seed_int = int(seed_hash, 16)
    rand_instance = random.Random(seed_int)
    return rand_instance


class Semiring(ABC):
    """
    Abstract base class representing a semiring.

    A semiring defines two operations, addition and multiplication, along with
    their identity elements. Subclasses must implement these methods.
    """

    @abstractmethod
    def zero(self):
        """Returns the additive identity element."""
        pass

    @abstractmethod
    def one(self):
        """Returns the multiplicative identity element."""
        pass

    @abstractmethod
    def add(self, a, b):
        """Defines the addition operation."""
        pass

    @abstractmethod
    def mul(self, a, b):
        """Defines the multiplication operation."""
        pass

    @abstractmethod
    def get_provenance_token(self, table_name: str, idx: int):
        """Returns a token for the given table name and tuple index."""
        pass

    @abstractmethod
    def convert_provenance(self, provenance):
        """Converts the provenance token to the semiring's format."""
        pass


class NaturalNumbersSemiring(Semiring):
    """
    Semiring over the natural numbers with standard addition and multiplication.
    """

    def zero(self) -> int:
        return 0

    def one(self) -> int:
        return 1

    def add(self, a: int, b: int) -> int:
        return a + b

    def mul(self, a: int, b: int) -> int:
        return a * b

    def get_provenance_token(self, table_name: str, idx: int) -> int:
        rand = generate_deterministic_random(table_name, idx)
        # Generate a random natural number between 1 and 100
        return rand.randint(1, 100)

    def convert_provenance(self, x: Any) -> int:
        value = int(x)
        assert value >= 0, f"Invalid natural number: {x}"
        return value


class TropicalCostSemiring(Semiring):
    """
    Semiring used in optimization problems, with addition as minimum and multiplication as addition.
    """

    def zero(self) -> float:
        return float("inf")  # Additive identity (maximum cost)

    def one(self) -> float:
        return 0.0  # Multiplicative identity (no cost)

    def add(self, a: float, b: float) -> float:
        return min(a, b)

    def mul(self, a: float, b: float) -> float:
        return a + b

    def get_provenance_token(self, table_name: str, idx: int) -> float:
        rand = generate_deterministic_random(table_name, idx)
        # Generate a random cost value between 1 and 100
        return float(rand.randint(1, 100))

    def convert_provenance(self, x: Any) -> float:
        return float(x)


class AccessLevel:
    """
    Represents an access control level.

    Levels are ordered from most to least restrictive:
    'TS' (Top Secret), 'S' (Secret), 'C' (Confidential), 'P' (Private), 'O' (Open).
    """

    levels = ["TS", "S", "C", "P", "O"]

    def __init__(self, level: str):
        if level not in self.levels:
            raise ValueError(f"Invalid access level: {level}")
        self.level = level

    def __repr__(self):
        return f"AccessLevel('{self.level}')"


class AccessControlSemiring(Semiring):
    """
    Semiring for access control, where addition is the minimum (least restrictive level),
    and multiplication is the maximum (most restrictive level).
    """

    def zero(self) -> AccessLevel:
        return AccessLevel(AccessLevel.levels[-1])  # Additive identity (least restrictive)

    def one(self) -> AccessLevel:
        return AccessLevel(AccessLevel.levels[0])  # Multiplicative identity (most restrictive)

    def add(self, a: AccessLevel, b: AccessLevel) -> AccessLevel:
        # Choose the less restrictive level (minimum)
        if not (isinstance(a, AccessLevel) and isinstance(b, AccessLevel)):
            return float("NaN")
        return a if AccessLevel.levels.index(a.level) > AccessLevel.levels.index(b.level) else b

    def mul(self, a: AccessLevel, b: AccessLevel) -> AccessLevel:
        # Choose the more restrictive level (maximum)
        if not (isinstance(a, AccessLevel) and isinstance(b, AccessLevel)):
            return float("NaN")
        return a if AccessLevel.levels.index(a.level) < AccessLevel.levels.index(b.level) else b

    def get_provenance_token(self, table_name: str, idx: int) -> AccessLevel:
        rand = generate_deterministic_random(table_name, idx)
        level = rand.choice(AccessLevel.levels)
        return AccessLevel(level=level)

    def convert_provenance(self, x: Any) -> AccessLevel:
        # Ensure the provenance is an AccessLevel instance with a valid level
        if isinstance(x, AccessLevel):
            assert x.level in AccessLevel.levels, f"Invalid AccessLevel: {x.level}"
            return x
        else:
            assert str(x) in AccessLevel.levels, f"Invalid AccessLevel: {x}"
            return AccessLevel(level=str(x))


class PolynomialSemiring(Semiring):
    """
    Semiring of polynomials used for provenance tracking, with symbolic expressions.
    """

    def zero(self):
        return 0

    def one(self):
        return 1

    def add(self, a, b):
        return sympy.Add(a, b, evaluate=False)

    def mul(self, a, b):
        return sympy.Mul(a, b, evaluate=False)

    def get_provenance_token(self, table_name: str, idx: int):
        # Generate a unique symbol for each row
        return sympy.Symbol(f"X_{table_name}_{idx}")

    def convert_provenance(self, x):
        # Convert existing provenance to a SymPy Symbol if it's not already
        if isinstance(x, sympy.Symbol):
            return x
        else:
            return sympy.Symbol(str(x))


class BooleanSemiring(Semiring):
    """
    Semiring over booleans with logical OR as addition and logical AND as multiplication.
    """

    def zero(self) -> bool:
        return False

    def one(self) -> bool:
        return True

    def add(self, a: bool, b: bool) -> bool:
        return a or b

    def mul(self, a: bool, b: bool) -> bool:
        return a and b

    def get_provenance_token(self, table_name: str, idx: int) -> bool:
        rand = generate_deterministic_random(table_name, idx)
        return rand.choice([True, False])

    def convert_provenance(self, x: Any) -> bool:
        return bool(x)


class BooleanPolynomialSemiring(Semiring):
    """
    Semiring that represents boolean expressions using symbolic logic operations.
    """

    def zero(self):
        return sympy.false

    def one(self):
        return sympy.true

    def add(self, a, b):
        if isna(a) or isna(b):
            return float("NaN")
        return sympy.Or(a, b)

    def mul(self, a, b):
        if isna(a) or isna(b):
            return float("NaN")
        return sympy.And(a, b)

    def get_provenance_token(self, table_name: str, idx: int):
        # Generate a unique symbol for each row
        return sympy.Symbol(f"X_{table_name}_{idx}")

    def convert_provenance(self, x):
        # Convert existing provenance to a SymPy Symbol if it's not already
        if isinstance(x, sympy.Symbol):
            return x
        else:
            return sympy.Symbol(str(x))
