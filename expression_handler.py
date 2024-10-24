"""
This module handles the combination of provenance expressions using semiring operations.
"""

from functools import reduce
from typing import Any, List
from consts import ADD, MUL
from semiring import Semiring


class ExpressionHandler:
    """
    Handles the combination of provenance expressions using semiring operations.

    Attributes:
        semiring (Semiring): The semiring instance that defines the operations.
    """

    def __init__(self, semiring: Semiring):
        self.semiring = semiring

    def combine_provenance(self, expressions: List[Any], operation: str) -> Any:
        """
        Combines a list of provenance expressions using the specified semiring operation.

        Args:
            expressions (List[Any]): A list of provenance expressions to combine.
            operation (str): The operation to use for combining the expressions.
                - 'add': Combine expressions using the semiring addition.
                - 'mul': Combine expressions using the semiring multiplication.

        Returns:
            Any: The combined provenance expression.

        Raises:
            ValueError: If an unsupported operation is specified.
        """
        if operation == MUL:
            # Multiply expressions using semiring multiplication
            return reduce(self.semiring.mul, expressions)
        elif operation == ADD:
            # Add expressions using semiring addition
            return reduce(self.semiring.add, expressions)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
