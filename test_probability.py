import time
import pandas as pd
import pytest
from consts import PROVENANCE
from probability_calc import compute_probability, replace_operations
from query_processor import QueryProcessor
from semiring import BooleanPolynomialSemiring, PolynomialSemiring
from sympy import Symbol  # type: ignore
from sympy.logic.boolalg import Boolean  # type: ignore


@pytest.fixture(
    params=[
        PolynomialSemiring(),
        BooleanPolynomialSemiring(),
    ],
)
def qp(request: pytest.FixtureRequest) -> QueryProcessor:
    """Initialize QueryProcessor and load test data into it."""

    # Initialize QueryProcessor
    qp = QueryProcessor(semiring=request.param)

    # Toy data setup
    students_data = [
        {"student_id": 1, "name": "Alice", "age": 20, "major": "Physics", "probability": 0.9},
        {"student_id": 2, "name": "Bob", "age": 22, "major": "Chemistry", "probability": 0.8},
        {"student_id": 3, "name": "Charlie", "age": 21, "major": "Physics", "probability": 0.85},
    ]
    students_df = pd.DataFrame(students_data)

    courses_data = [
        {"course_id": 101, "title": "Mechanics", "department": "Physics", "probability": 0.95},
        {
            "course_id": 201,
            "title": "Organic Chemistry",
            "department": "Chemistry",
            "probability": 0.9,
        },
    ]
    courses_df = pd.DataFrame(courses_data)

    enrollments_data = [
        {"student_id": 1, "course_id": 201, "probability": 0.88},
        {"student_id": 2, "course_id": 201, "probability": 0.92},
        {"student_id": 3, "course_id": 101, "probability": 0.9},
    ]
    enrollments_df = pd.DataFrame(enrollments_data)

    # Load tables into QueryProcessor and assign provenance and probabilities
    qp.load_table(df=students_df, table_name="students", probability_col_name="probability")
    qp.load_table(df=courses_df, table_name="courses", probability_col_name="probability")
    qp.load_table(df=enrollments_df, table_name="enrollments", probability_col_name="probability")

    return qp


def test_provenance_probability_mapping(qp: QueryProcessor):
    """Test that the provenance-probability mappings are correctly created for each table."""
    print("\n=== Testing Provenance-Probability Mapping ===")

    # Check that the mappings exist for each table
    for table_name in ["students", "courses", "enrollments"]:
        df = qp.tables[table_name]

        # Ensure that every provenance symbol in the DataFrame has a corresponding probability
        for _, row in df.iterrows():
            provenance_symbol = row[PROVENANCE]
            assert isinstance(provenance_symbol, Symbol) or isinstance(
                provenance_symbol, Boolean
            ), f"Provenance symbol {provenance_symbol} for table '{table_name}' is not a Symbol."
            assert provenance_symbol in qp.provenance_probability_map, (
                f"Provenance symbol {provenance_symbol} not found in mapping for table"
                f" '{table_name}'."
            )

            probability = qp.provenance_probability_map[provenance_symbol]
            assert (
                0 <= probability <= 1
            ), f"Probability {probability} for symbol {provenance_symbol} is not between 0 and 1."

    print("Provenance-probability mappings are valid for all tables.")


def test_probability_with_random_generation(qp: QueryProcessor):
    """Test that random probabilities are generated when no probability column is provided."""
    print("\n=== Testing Random Probability Generation ===")

    # Load a new table without probability column
    grades_data = [
        {"student_id": 1, "course_id": 201, "grade": "A"},
        {"student_id": 2, "course_id": 201, "grade": "B"},
        {"student_id": 3, "course_id": 101, "grade": "A"},
    ]
    grades_df = pd.DataFrame(grades_data)

    qp.load_table(df=grades_df, table_name="grades")
    grades_provenance = qp.tables["grades"][PROVENANCE]

    # Ensure that probabilities are between 0 and 1
    for provenance_symbol in grades_provenance:
        assert (
            provenance_symbol in qp.provenance_probability_map
        ), f"Provenance symbol {provenance_symbol} not found in mapping for table 'grades'."
        probability = qp.provenance_probability_map[provenance_symbol]
        assert (
            0 <= probability <= 1
        ), f"Randomly generated probability {probability} is not between 0 and 1."
        print(f"Provenance symbol: {provenance_symbol}, Random probability: {probability}")

    print("Random probability generation test passed.")


@pytest.mark.parametrize("exact_computation", [True, False])
def test_compute_probability(qp: QueryProcessor, exact_computation):
    """Test the compute_probability function using provenance expressions from the query results."""
    print(f"\n=== Testing Compute Probability (exact_computation={exact_computation}) ===")

    # Step 1: Perform a query (join between students and enrollments)
    students_df = qp.tables["students"]
    enrollments_df = qp.tables["enrollments"]
    # Perform the join operation
    joined_df = qp.join(left=students_df, right=enrollments_df, on=["student_id"])
    print(f"Joined data:\n{joined_df}\n")

    # Step 2: Project over 'major'
    projected_df = qp.project(table=joined_df, columns=["major"])
    print(f"Projected data:\n{projected_df}\n")

    # Step 4: Compute the probability for each unique 'major'
    for _, row in projected_df.iterrows():
        major = row["major"]
        provenance_expr = row[PROVENANCE]
        print(f"Computing probability for major '{major}' with provenance '{provenance_expr}'")

        # Use the compute_probability function to compute the probability
        final_probability = compute_probability(
            provenance_expr,
            qp.provenance_probability_map,
            qp.semiring,
            exact_computation=exact_computation,
        )
        print(f"Computed probability for major '{major}': {final_probability:.8f}")

        # Step 5: Compute the expected probability manually
        if major == "Chemistry":
            # Expected probability is P(X_enrollments_1) * P(X_students_1)
            expected_probability = (
                qp.provenance_probability_map[Symbol("X_enrollments_1")]
                * qp.provenance_probability_map[Symbol("X_students_1")]
            )
            print(f"Expected probability for 'Chemistry': {expected_probability:.8f}")
        elif major == "Physics":
            # Expected probability is:
            # P(X_enrollments_0) * P(X_students_0) + P(X_enrollments_2) * P(X_students_2)
            # Use inclusion-exclusion principle since events are independent but not mutually exclusive
            term1_prob = (
                qp.provenance_probability_map[Symbol("X_enrollments_0")]
                * qp.provenance_probability_map[Symbol("X_students_0")]
            )
            term2_prob = (
                qp.provenance_probability_map[Symbol("X_enrollments_2")]
                * qp.provenance_probability_map[Symbol("X_students_2")]
            )
            expected_probability = term1_prob + term2_prob - (term1_prob * term2_prob)
            print(f"Expected probability for 'Physics': {expected_probability:.8f}")
        else:
            raise ValueError(f"Unexpected major '{major}' in data.")

        # Step 6: Assert that the computed probability matches the expected probability
        if exact_computation:
            # For exact computation, use a small tolerance
            tolerance = 1e-6
        else:
            # For approximate computation, allow a larger tolerance due to sampling error
            tolerance = 1e-2  # 1%

        assert (
            abs(final_probability - expected_probability) < tolerance
        ), f"Probability mismatch for major '{major}' with exact_computation={exact_computation}"

    print(f"\nCompute probability test passed for exact_computation={exact_computation}.")


import pytest
import random
import sympy
from sympy import Symbol
from typing import List, Dict


def test_large_dnf_expression():
    """Test compute_probability function with a large DNF expression."""
    print("\n=== Testing Compute Probability with Large DNF Expression ===")

    # Seed for reproducibility
    random.seed(42)

    # Step 1: Generate variables and probabilities
    num_variables = 10
    variables = [Symbol(f"X_{i}") for i in range(num_variables)]
    probabilities = {var: random.uniform(0.1, 0.9) for var in variables}

    # Step 2: Construct the DNF expression
    num_terms = num_variables // 2  # Number of terms in the DNF expression
    terms = []
    used_variables = set()

    for _ in range(num_terms):
        # Randomly select a subset of variables for this term
        available_variables = list(set(variables) - used_variables)
        term_size = random.randint(1, 3)  # Number of variables in the term
        if len(available_variables) < term_size:
            # If not enough variables left, break
            break
        term_vars = random.sample(available_variables, term_size)
        # Add the variables to used_variables to ensure mutual exclusivity
        used_variables.update(term_vars)
        # Create a logical AND of the selected variables
        term_expr = sympy.And(*term_vars)
        terms.append(term_expr)

    # Step 3: Build the overall DNF expression
    dnf_expr = sympy.Or(*terms)

    # Ensure expected_probability does not exceed 1 due to floating point arithmetic
    start_time = time.time()
    expected_probability = compute_probability(
        dnf_expr,
        probabilities,
        BooleanPolynomialSemiring(),
        exact_computation=True,
    )
    end_time = time.time()
    print(f"Time taken for exact computation: {end_time - start_time:.4f} seconds")

    print(f"Expected probability (analytical): {expected_probability:.8f}")

    start_time = time.time()
    estimated_probability = compute_probability(
        dnf_expr,
        probabilities,
        BooleanPolynomialSemiring(),
        exact_computation=False,
    )
    end_time = time.time()
    print(f"Time taken for Monte Carlo simulation: {end_time - start_time:.4f} seconds")
    print(f"Estimated probability (Monte Carlo): {estimated_probability:.8f}")

    # Step 6: Compare probabilities
    tolerance = 1e-2  # Allow 1% difference due to sampling error
    assert (
        abs(estimated_probability - expected_probability) < tolerance
    ), f"Probability mismatch: expected {expected_probability}, estimated {estimated_probability}"

    print("\nCompute probability test passed for large DNF expression.")


def test_replace_operations():
    """
    Test the replace_operations function to ensure that it correctly replaces Add with Or
    and Mul with And in SymPy expressions.
    """
    a, b, c = sympy.symbols("a b c")

    # Test case 1: Simple expression with Mul inside Add
    expr1 = sympy.Add(sympy.Mul(a, b, evaluate=False), c, evaluate=False)  # (a * b) + c
    result1 = replace_operations(expr1)
    expected1 = sympy.Or(sympy.And(a, b), c)  # (a & b) | c
    assert result1 == expected1, f"Expected {expected1}, but got {result1}"

    # Test case 2: Nested expressions
    expr2 = sympy.Add(
        sympy.Mul(a, sympy.Add(b, c, evaluate=False), evaluate=False), c, evaluate=False
    )  # a * (b + c) + c
    result2 = replace_operations(expr2)
    expected2 = sympy.Or(sympy.And(a, sympy.Or(b, c)), c)  # (a & (b | c)) | c
    assert result2 == expected2, f"Expected {expected2}, but got {result2}"

    # Test case 3: Expression without any Add or Mul
    expr3 = a
    result3 = replace_operations(expr3)
    expected3 = a  # No changes
    assert result3 == expected3, f"Expected {expected3}, but got {result3}"

    # Test case 4: More complex nested expression
    expr4 = sympy.Add(
        sympy.Mul(a, b, evaluate=False), sympy.Mul(b, c, evaluate=False), evaluate=False
    )  # (a * b) + (b * c)
    result4 = replace_operations(expr4)
    expected4 = sympy.Or(sympy.And(a, b), sympy.And(b, c))  # (a & b) | (b & c)
    assert result4 == expected4, f"Expected {expected4}, but got {result4}"

    # Test case 5: Testing simplify on logical formula
    expr5 = sympy.Add(sympy.Add(a, b, evaluate=False), c, evaluate=False)  # a + b + c
    add_expr5 = sympy.Add(expr5, expr5, evaluate=False)  # (a + b + c) + (a + b + c)
    mul_add_expr5 = sympy.Mul(add_expr5, expr5, evaluate=False)
    simply_expr5 = sympy.simplify(replace_operations(expr5))
    simply_mul_add_expr5 = sympy.simplify(replace_operations(mul_add_expr5))
    print(f"simply_expr5: {simply_expr5}")
    print(f"simply_mul_add_expr5 before replacing: {sympy.simplify(mul_add_expr5)}")
    print(f"simply_mul_add_expr5 after replacing: {simply_mul_add_expr5}")
    assert simply_expr5 == simply_mul_add_expr5
    print("All replace_operations test cases passed.")
