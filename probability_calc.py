"""
This module provides functions to compute the probability of provenance expressions
represented as SymPy expressions, using exact or approximate methods.
"""

import random
from typing import Dict, Tuple
import sympy  # type: ignore
from semiring import BooleanPolynomialSemiring, PolynomialSemiring, Semiring


def replace_operations(expr):
    """
    Recursively replaces Add and Mul operations in a SymPy expression with logical Or and And, respectively.

    Args:
        expr (sympy.Basic): The SymPy expression to transform. It can contain arithmetic Add and Mul operations.

    Returns:
        sympy.Basic: A new SymPy expression where:
                     - Add operations are replaced with logical Or.
                     - Mul operations are replaced with logical And.
                     - Any other terms (symbols, constants) remain unchanged.

    Example:
        >>> a, b, c = sympy.symbols('a b c')
        >>> expr = sympy.Add(sympy.Mul(a, b, evaluate=False), c, evaluate=False)
        >>> replace_operations(expr)
        (a & b) | c
    """
    if isinstance(expr, sympy.Add):
        return sympy.Or(*[replace_operations(arg) for arg in expr.args])
    elif isinstance(expr, sympy.Mul):
        return sympy.And(*[replace_operations(arg) for arg in expr.args])
    else:
        return expr


def preprocess_expression_and_probabilities(
    provenance_expr: sympy.Expr, probabilities: Dict[sympy.Symbol, float], semiring: Semiring
) -> Tuple[sympy.Expr, Dict[sympy.Symbol, float]]:
    """
    Preprocesses the provenance expression and probabilities based on the semiring.

    For the PolynomialSemiring:
    - Replaces arithmetic operations with logical operations.
    - Simplifies the expression.

    For the BooleanPolynomialSemiring:
    - Assumes the expression is already a boolean expression.

    Args:
        provenance_expr (sympy.Expr): The provenance expression to preprocess.
        probabilities (Dict[sympy.Symbol, float]): A dictionary mapping symbols to their probabilities.
        semiring (Semiring): The semiring used in the computation.

    Returns:
        Tuple[sympy.Expr, Dict[sympy.Symbol, float]]: The preprocessed boolean expression and the adjusted probabilities dictionary.

    Raises:
        ValueError: If the semiring type is unsupported.
        AssertionError: If probabilities for symbols are missing.
    """
    if isinstance(semiring, PolynomialSemiring):
        bool_expr = sympy.simplify(replace_operations(provenance_expr))

    elif isinstance(semiring, BooleanPolynomialSemiring):
        # Expression is already a boolean expression
        bool_expr = provenance_expr
    else:
        raise ValueError("Unsupported semiring type.")

    # Ensure all symbols have associated probabilities
    symbols_in_expression = bool_expr.atoms(sympy.Symbol)
    missing_symbols = symbols_in_expression - probabilities.keys()
    assert len(missing_symbols) == 0, f"Missing probabilities for symbols: {missing_symbols}"
    # Adjust probabilities to include only relevant symbols
    probabilities = {s: probabilities[s] for s in symbols_in_expression}

    return bool_expr, probabilities


def compute_probability(
    provenance_expr: sympy.Expr,
    probabilities: Dict[sympy.Symbol, float],
    semiring: Semiring,
    exact_computation: bool = True,
) -> float:
    """
    Computes the probability of a provenance expression represented as a SymPy boolean expression.

    The function performs the following steps:
    1. Preprocesses the provenance expression and probabilities based on the semiring.
    2. Computes the probability of the boolean expression.
        - If exact_computation is True, uses a recursive exact method.
        - If exact_computation is False, uses Monte Carlo sampling to estimate the probability.

    Args:
        provenance_expr (sympy.Expr): A SymPy expression representing the provenance.
        probabilities (Dict[sympy.Symbol, float]): A dictionary mapping SymPy symbols to their probabilities.
        semiring (Semiring): The semiring used in the computation.
        exact_computation (bool): If True, performs exact computation using recursion. If False, uses Monte Carlo sampling.

    Returns:
        float: The computed probability of the provenance expression.
    """
    # Preprocess the expression and probabilities
    bool_expr, probabilities = preprocess_expression_and_probabilities(
        provenance_expr, probabilities, semiring
    )

    if exact_computation:
        # Compute the probability recursively
        return compute_prob_recursive(bool_expr, probabilities)
    else:
        # Estimate the probability using Monte Carlo sampling
        return compute_prob_monte_carlo(bool_expr, probabilities)


def compute_prob_recursive(
    expression: sympy.Expr, probabilities: Dict[sympy.Symbol, float]
) -> float:
    """
    Recursively computes the probability of a boolean expression represented as a SymPy logical expression.

    The function applies the probability formula for Boolean expressions:
    P(expr) = P(var) * P(expr | var=True) + (1 - P(var)) * P(expr | var=False)

    Args:
        expression (sympy.Expr): The boolean expression to compute the probability of.
        probabilities (Dict[sympy.Symbol, float]): A dictionary mapping symbols to their probabilities.

    Returns:
        float: The computed probability.
    """
    # Simplify the expression if possible
    expression = sympy.simplify_logic(expression)

    # Base cases
    if expression == sympy.true:
        return 1.0
    if expression == sympy.false:
        return 0.0

    # Get the set of symbols in the expression
    symbols = expression.atoms(sympy.Symbol)
    if not symbols:
        # The expression is a constant (True or False)
        return float(expression)

    # Select a variable to assign
    var = next(iter(symbols))

    # Evaluate the expression with var = True
    expr_true = expression.subs(var, True)
    p_true = compute_prob_recursive(expr_true, probabilities)

    # Evaluate the expression with var = False
    expr_false = expression.subs(var, False)
    p_false = compute_prob_recursive(expr_false, probabilities)

    # Compute the total probability
    p_var = probabilities[var]
    return p_var * p_true + (1 - p_var) * p_false


def compute_prob_monte_carlo(
    expression: sympy.Expr, probabilities: Dict[sympy.Symbol, float], num_samples: int = 10000
) -> float:
    """
    Estimates the probability of a boolean expression using Monte Carlo sampling.

    Args:
        expression (sympy.Expr): The boolean expression to estimate the probability of.
        probabilities (Dict[sympy.Symbol, float]): A dictionary mapping symbols to their probabilities.
        num_samples (int): The number of samples to use in the estimation.

    Returns:
        float: The estimated probability.
    """
    symbols = list(expression.atoms(sympy.Symbol))
    true_count = 0

    for _ in range(num_samples):
        # Generate a random assignment for each symbol
        assignment = {symbol: random.random() < probabilities[symbol] for symbol in symbols}

        # Evaluate the expression with the assignment
        result = expression.subs(assignment)

        if result == sympy.true:
            true_count += 1
        elif result == sympy.false:
            pass  # result is False
        else:
            # Should not happen
            raise ValueError(f"Unexpected result during evaluation: {result}")

    estimated_probability = true_count / num_samples
    return estimated_probability
