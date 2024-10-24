import pandas as pd
import pytest
from consts import BAG, PROVENANCE, SET
from query_processor import QueryProcessor
import sympy  # type: ignore
from functools import reduce

from semiring import (
    AccessControlSemiring,
    BooleanPolynomialSemiring,
    BooleanSemiring,
    NaturalNumbersSemiring,
    PolynomialSemiring,
    TropicalCostSemiring,
)
from utils import load_custom_data


@pytest.fixture(
    params=[
        NaturalNumbersSemiring(),
        PolynomialSemiring(),
        TropicalCostSemiring(),
        BooleanSemiring(),
        AccessControlSemiring(),
        BooleanPolynomialSemiring(),
    ],
)
def qp(request: pytest.FixtureRequest) -> QueryProcessor:
    """Initialize QueryProcessor and load test data into it."""

    # Initialize QueryProcessor
    qp = QueryProcessor(semiring=request.param)

    # Toy data setup
    students_data = [
        {"student_id": 1, "name": "Alice", "age": 20, "major": "Physics"},
        {"student_id": 1, "name": "Alice", "age": 20, "major": "Physics"},
        {"student_id": 2, "name": "Bob", "age": 22, "major": "Chemistry"},
        {"student_id": 3, "name": "Charlie", "age": 21, "major": "Physics"},
        {"student_id": 4, "name": "David", "age": 23, "major": "Biology"},
        {"student_id": 5, "name": "Eve", "age": 20, "major": "Physics"},
    ]
    students_df = pd.DataFrame(students_data)

    courses_data = [
        {"course_id": 101, "title": "Mechanics", "department": "Physics"},
        {"course_id": 102, "title": "Thermodynamics", "department": "Physics"},
        {"course_id": 201, "title": "Organic Chemistry", "department": "Chemistry"},
        {"course_id": 301, "title": "Genetics", "department": "Biology"},
    ]
    courses_df = pd.DataFrame(courses_data)

    enrollments_data = [
        {"student_id": 1, "course_id": 201},
        {"student_id": 2, "course_id": 201},
        {"student_id": 3, "course_id": 101},
        {"student_id": 3, "course_id": 102},
        {"student_id": 3, "course_id": 102},
        {"student_id": 4, "course_id": 301},
        {"student_id": 5, "course_id": 102},
    ]
    enrollments_df = pd.DataFrame(enrollments_data)

    # Load tables into QueryProcessor and assign provenance
    qp.load_table(df=students_df, table_name="students")
    qp.load_table(df=courses_df, table_name="courses")
    qp.load_table(df=enrollments_df, table_name="enrollments")

    return qp


@pytest.fixture(
    params=[
        PolynomialSemiring(),
        BooleanPolynomialSemiring(),
    ],
)
def olympics_qp(request: pytest.FixtureRequest):
    """
    Initialize QueryProcessor with PolynomialSemiring and load Olympics data.
    """
    qp = QueryProcessor(semiring=request.param)
    load_custom_data(qp, "olympic_historical_dataset")

    return qp


def test_default_dataset_ops(olympics_qp: QueryProcessor) -> None:
    """
    Test selecting the city 'Paris' from the Olympics_Games table
    and compute the provenance probabilities.
    Test the default dataset with few operations.
    """
    # Project the 'city' column from Olympics_Games
    games_city_proj = olympics_qp.project(table="Olympics_Games", columns=["city"])

    # Select rows where 'city' is 'Paris'
    prov = olympics_qp.select(table=games_city_proj, condition="city == 'Paris'").iloc[0][
        "provenance"
    ]

    # Validate provenance
    assert prov is not None, "Provenance should not be None"

    # Print provenance atoms and their probabilities
    cum_prod = 1.0
    for atom in prov.atoms():
        print(f"{atom=}, prob={olympics_qp.provenance_probability_map[atom]}")
        cum_prod *= 1 - olympics_qp.provenance_probability_map[atom]
    expected_prob = 1 - cum_prod
    # Compute exact and Monte Carlo probabilities
    exact_prob = olympics_qp.compute_probability(provenance_expr=prov)
    monte_carlo_prob = olympics_qp.compute_probability(
        provenance_expr=prov, exact_computation=False
    )

    # Validate probabilities
    assert exact_prob > 0, "Exact probability should be greater than 0"
    assert monte_carlo_prob > 0, "Monte Carlo probability should be greater than 0"
    assert abs(exact_prob - monte_carlo_prob) < 0.05, "The probabilities should be close"
    assert abs(exact_prob - expected_prob) < 0.0001, "The probabilities should be close"

    print(f"Exact probability: {exact_prob}")
    print(f"Monte Carlo probability: {monte_carlo_prob}")


def test_selection(qp: QueryProcessor) -> None:
    """Test selection with set and bag semantics."""
    print("=== Testing Selection ===")

    ### Test Selection (Set Semantics) ###
    selected_df = qp.select(table="students", condition="major == 'Physics'")
    print(selected_df)
    assert len(selected_df) == 3, f"Expected 3 rows, but got {len(selected_df)}"

    for name in selected_df["name"].unique():
        provenance_set = selected_df[selected_df["name"] == name][PROVENANCE].values[0]
        provenance_bag = qp.tables["students"][
            (qp.tables["students"]["name"] == name) & (qp.tables["students"]["major"] == "Physics")
        ][PROVENANCE]
        expected_provenance = reduce(qp.semiring.add, provenance_bag)
        assert provenance_set == expected_provenance, f"Provenance mismatch for {name}"

    print("Set semantics selection test passed.")

    ### Test Selection (Bag Semantics) ###
    qp.semantics = BAG
    selected_df_bag = qp.select(table="students", condition="major == 'Physics'")
    print(selected_df_bag)
    assert len(selected_df_bag) == 4, f"Expected 4 rows, but got {len(selected_df_bag)}"

    for name in selected_df_bag["name"].unique():
        provenance_bag = selected_df_bag[selected_df_bag["name"] == name][PROVENANCE].values
        provenance_original = qp.tables["students"][
            (qp.tables["students"]["name"] == name) & (qp.tables["students"]["major"] == "Physics")
        ][PROVENANCE].values
        assert (provenance_bag == provenance_original).all(), f"Provenance mismatch for {name}"

    print("Bag semantics selection test passed.")


def test_projection(qp: QueryProcessor) -> None:
    """Test projection with set and bag semantics."""
    print("\n=== Testing Projection ===")

    students_df = qp.tables["students"]

    # Set semantics
    projected_df = qp.project(table=students_df, columns=["name", "age"])
    print(projected_df)
    assert len(projected_df) == len(students_df) - 1, (
        f"Expected {len(students_df) - 1} rows in projected set semantics, but got"
        f" {len(projected_df)}"
    )
    print("Set semantics projection test passed.")

    # Bag semantics
    qp.semantics = BAG
    projected_df_bag = qp.project(table=students_df, columns=["name", "age"])
    print(projected_df_bag)
    assert len(projected_df_bag) == len(students_df), (
        f"Expected {len(students_df)} rows in projected bag semantics, but got"
        f" {len(projected_df_bag)}"
    )
    print("Bag semantics projection test passed.")

    for name in students_df["name"].unique():
        provenance_set = projected_df[projected_df["name"] == name][PROVENANCE].values[0]
        provenance_bag = projected_df_bag[projected_df_bag["name"] == name][PROVENANCE].values
        expected_provenance = qp.tables["students"][qp.tables["students"]["name"] == name][
            PROVENANCE
        ]
        assert provenance_set == reduce(
            qp.semiring.add, expected_provenance
        ), f"Provenance mismatch for {name}"
        assert (provenance_bag == expected_provenance).all(), f"Provenance mismatch for {name}"


def test_join(qp: QueryProcessor) -> None:
    """Test join with set and bag semantics."""
    print("\n=== Testing Join ===")

    # Set semantics
    selected_df = qp.select(table="students", condition="major == 'Physics'")
    print(f"Selected students:\n{selected_df}")
    enrollments_df = qp.tables["enrollments"]
    print(f"Enrollments:\n{enrollments_df}")
    join_df = qp.join(left=selected_df, right=enrollments_df, on=["student_id"])
    print(join_df)
    assert len(join_df) == 4, f"Expected 4 rows in join with set semantics, but got {len(join_df)}"
    print("Set semantics join test passed.")

    # Bag semantics
    qp.semantics = BAG
    join_df_bag = qp.join(left=selected_df, right=enrollments_df, on=["student_id"])
    print(join_df_bag)
    assert (
        len(join_df_bag) == 5
    ), f"Expected 5 rows in join with bag semantics, but got {len(join_df_bag)}"
    print("Bag semantics join test passed.")

    # test cross
    qp.semantics = SET
    courses_df = qp.tables["courses"]
    cross_df = qp.join(left=enrollments_df, right=courses_df, how="cross")
    print(cross_df)

    assert len(cross_df) == len(enrollments_df) * len(courses_df) - len(courses_df), (
        f"Expected {len(enrollments_df) * len(courses_df) - len(courses_df)} rows in cross join,"
        f" but got {len(cross_df)}"
    )


def test_union(qp: QueryProcessor) -> None:
    """Test union operation."""
    print("\n=== Testing Union ===")

    # Set semantics
    project_courses_df = qp.project(table=qp.tables["courses"], columns=["course_id"])
    print(f"Projected courses:\n{project_courses_df}")

    project_enrollments_df = qp.project(table=qp.tables["enrollments"], columns=["course_id"])
    print(f"Projected enrollments:\n{project_enrollments_df}")

    union_df = qp.union(table1=project_courses_df, table2=project_enrollments_df)
    print(f"Union:\n{union_df}")

    for course_id in union_df["course_id"].unique():
        provenance_set = union_df[union_df["course_id"] == course_id][PROVENANCE].values[0]
        provenance_courses = qp.tables["courses"][qp.tables["courses"]["course_id"] == course_id][
            PROVENANCE
        ]
        provenance_enrollments = qp.tables["enrollments"][
            qp.tables["enrollments"]["course_id"] == course_id
        ][PROVENANCE]
        expected_provenance = qp.semiring.add(
            reduce(qp.semiring.add, provenance_courses),
            reduce(qp.semiring.add, provenance_enrollments),
        )
        if isinstance(expected_provenance, sympy.Expr):
            assert sympy.simplify(provenance_set) == sympy.simplify(
                expected_provenance
            ), f"Provenance mismatch for {course_id}"
        else:
            assert provenance_set == expected_provenance, f"Provenance mismatch for {course_id}"

    # Check that the union removed duplicates
    assert len(union_df) == len(
        set(
            list(project_courses_df["course_id"].unique())
            + list(project_enrollments_df["course_id"].unique())
        )
    ), f"Expected 3 rows in union, but got {len(union_df)}"

    # Bag semantics
    qp.semantics = BAG
    union_df_bag = qp.union(table1=project_courses_df, table2=project_enrollments_df)
    print(f"Union (bag semantics):\n{union_df_bag}")
    assert len(union_df_bag) == len(project_courses_df) + len(project_enrollments_df), (
        f"Expected {len(project_courses_df) + len(project_enrollments_df)} rows in union, but got"
        f" {len(union_df_bag)}"
    )

    print("Union test passed.")


def test_intersection(qp: QueryProcessor) -> None:
    """Test intersection operation."""
    print("\n=== Testing Intersection ===")

    # Select students in Physics and Chemistry
    physics_students = qp.select(table="students", condition="major == 'Physics'")
    chemistry_students = qp.select(table="students", condition="major == 'Chemistry'")

    # Perform intersection (should return empty since no overlap of student_id and major)
    intersection_df = qp.intersection(physics_students, chemistry_students)
    print(f"Intersection of Physics and Chemistry students:\n{intersection_df}")
    assert (
        len(intersection_df) == 0
    ), f"Expected 0 rows in intersection, but got {len(intersection_df)}"
    print("Intersection test passed (no matching rows).")

    project_courses_df = qp.project(table="courses", columns=["course_id"])
    print(f"Projected courses:\n{project_courses_df}")

    project_enrollments_df = qp.project(table=qp.tables["enrollments"], columns=["course_id"])
    print(f"Projected enrollments:\n{project_enrollments_df}")

    intersection_df = qp.intersection(project_courses_df, project_enrollments_df)
    print(f"Intersection of projected physics students and all students:\n{intersection_df}")

    # Check that we get the correct number of rows (overlapping "student_id" and "age")
    expected_intersection_len = len(qp.tables["courses"].drop_duplicates(subset=["course_id"]))
    assert (
        len(intersection_df) == expected_intersection_len
    ), f"Expected {expected_intersection_len} rows in intersection, but got {len(intersection_df)}"

    # Check the provenance values
    for course_id in intersection_df["course_id"].unique():
        provenance_set = intersection_df[intersection_df["course_id"] == course_id][
            PROVENANCE
        ].values[0]
        expected_provenance = qp.semiring.mul(
            project_courses_df[project_courses_df["course_id"] == course_id][PROVENANCE].values[0],
            project_enrollments_df[project_enrollments_df["course_id"] == course_id][
                PROVENANCE
            ].values[0],
        )
        assert (
            provenance_set == expected_provenance
        ), f"Provenance mismatch for course_id {course_id}"

    print("Intersection test passed.")


def test_difference(qp: QueryProcessor) -> None:
    """Test difference operation."""
    print("\n=== Testing Difference ===")

    # Select students in Physics and Chemistry
    physics_students = qp.select(table="students", condition="major == 'Physics'")
    chemistry_students = qp.select(table="students", condition="major == 'Chemistry'")

    # Perform difference (Physics students not in Chemistry)
    difference_df = qp.difference(physics_students, chemistry_students)
    print(f"Difference (Physics students not in Chemistry):\n{difference_df}")

    # We should expect all Physics students in the result, as no Chemistry student should be subtracted
    assert len(difference_df) == len(
        physics_students
    ), f"Expected {len(physics_students)} rows in difference, but got {len(difference_df)}"

    # Now perform difference for students not in Chemistry (should return Physics students)
    difference_df = qp.difference(qp.tables["students"], chemistry_students)
    print(f"Difference (All students not in Chemistry):\n{difference_df}")

    # Ensure no Chemistry students are present in the result
    assert all(
        difference_df["major"] != "Chemistry"
    ), "Found a Chemistry student in the difference result."

    # Check provenance for the difference
    for student_id in difference_df["student_id"].unique():
        provenance_set = difference_df[difference_df["student_id"] == student_id][
            PROVENANCE
        ].values[0]
        expected_provenance = qp.tables["students"][
            qp.tables["students"]["student_id"] == student_id
        ][PROVENANCE]
        assert provenance_set == reduce(
            qp.semiring.add, expected_provenance
        ), f"Provenance mismatch for student_id {student_id}"

    print("Difference test passed.")


def test_rename(qp: QueryProcessor) -> None:
    """Test rename operation."""
    print("\n=== Testing Rename ===")

    # Rename columns in 'students' table
    renamed_df = qp.rename(table="students", columns={"name": "student_name", "age": "student_age"})
    print(f"Renamed students:\n{renamed_df}")

    # Check that the columns have been renamed
    assert "student_name" in renamed_df.columns, "Column 'student_name' not found after rename."
    assert "student_age" in renamed_df.columns, "Column 'student_age' not found after rename."
    assert "name" not in renamed_df.columns, "Column 'name' should not exist after rename."
    assert "age" not in renamed_df.columns, "Column 'age' should not exist after rename."

    # Check that the data and provenance remain the same
    original_df = qp.tables["students"]
    # Since we didn't change data, the number of rows should be the same
    assert len(renamed_df) == len(original_df), "Number of rows changed after rename."
    # Check that the provenance column is the same
    assert (
        renamed_df[PROVENANCE] == original_df[PROVENANCE]
    ).all(), "Provenance mismatch after rename."

    print("Rename test passed.")
