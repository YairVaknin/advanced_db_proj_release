"""
Interactive IPython CLI Tool for QueryProcessor.

This script allows you to load data into the QueryProcessor and start an interactive IPython shell for experimentation.
"""

import argparse
import pickle
import pandas as pd
from IPython import embed
from query_processor import QueryProcessor
import semiring as sr
from utils import load_custom_data


def load_test_data(qp: QueryProcessor):
    """
    Loads small toy tables into the QueryProcessor for testing purposes.

    Args:
        qp (QueryProcessor): The QueryProcessor instance to load data into.

    The function creates toy datasets for 'students', 'courses', and 'enrollments',
    and loads them into the QueryProcessor with assigned provenance and probabilities.
    """
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
    print("Test data loaded into QueryProcessor.")


def main():
    """
    Main function to parse command-line arguments and start the interactive IPython shell.
    """
    # Dynamically retrieve all subclasses of Semiring
    available_semirings = {cls.__name__: cls for cls in sr.Semiring.__subclasses__()}

    parser = argparse.ArgumentParser(description="Interactive IPython CLI Tool for QueryProcessor")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "default", "custom"],
        default="default",
        help="Mode of operation: 'test', 'default', or 'custom'. Default is 'default'.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the folder containing CSV files (used in 'custom' mode).",
    )
    parser.add_argument(
        "--semiring",
        type=str,
        choices=list(available_semirings.keys()),
        default="PolynomialSemiring",
        help=(
            f"Semiring to use: {', '.join(available_semirings.keys())}. Default is"
            " 'PolynomialSemiring'."
        ),
    )
    parser.add_argument(
        "--semantics",
        type=str,
        choices=["set", "bag"],
        default="set",
        help="Semantics to use: 'set' or 'bag'. Default is 'set'.",
    )
    parser.add_argument(
        "--load_file",
        type=str,
        default=None,
        help="Path to a saved QueryProcessor instance to load instead of initializing a new one.",
    )
    args = parser.parse_args()

    # Load an existing QueryProcessor instance if a file is provided
    if args.load_file:
        try:
            with open(args.load_file, "rb") as file:
                qp = pickle.load(file)
            print(f"QueryProcessor instance loaded from {args.load_file}.")
        except Exception as e:
            print(f"Failed to load QueryProcessor from {args.load_file}: {e}")
            return
    else:
        # Choose the semiring class based on the argument
        semiring_class = available_semirings.get(args.semiring)
        if semiring_class:
            semiring = semiring_class()
        else:
            print(f"Unsupported semiring: {args.semiring}")
            return

        # Initialize QueryProcessor with specified semantics
        qp = QueryProcessor(semiring=semiring, semantics=args.semantics)

        # Load data based on the selected mode
        if args.mode == "test":
            load_test_data(qp)
        elif args.mode == "default":
            load_custom_data(qp, "olympic_historical_dataset")
        elif args.mode == "custom":
            if args.path is None:
                print("Error: '--path' must be provided in 'custom' mode.")
                return
            load_custom_data(qp, args.path)
        else:
            print(f"Unsupported mode: {args.mode}")
            return

    # Start an IPython shell with the QueryProcessor and tables available

    logo = r"""
   ___                        ____                                           ____  _          _ _ 
  / _ \ _   _  ___ _ __ _   _|  _ \ _ __ ___   ___ ___  ___ ___  ___  _ __  / ___|| |__   ___| | |
 | | | | | | |/ _ \ '__| | | | |_) | '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__| \___ \| '_ \ / _ \ | |
 | |_| | |_| |  __/ |  | |_| |  __/| | | (_) | (_|  __/\__ \__ \ (_) | |     ___) | | | |  __/ | |
  \__\_\\__,_|\___|_|   \__, |_|   |_|  \___/ \___\___||___/___/\___/|_|    |____/|_| |_|\___|_|_|
                        |___/                                                                     
    """
    banner = f"{logo}\nWelcome to the Interactive QueryProcessor Shell!"
    banner += "\nYou can access the QueryProcessor instance using the variable 'qp'."
    banner += "\nAvailable tables are stored in 'qp.tables'."
    banner += f"\nAvailable table names: {qp.get_available_tables(ret_string=True)}\n"

    # Define the local namespace for the IPython shell
    local_ns = {
        "qp": qp,
        "pd": pd,
        # Import other modules or functions as needed
    }

    # Start the IPython shell
    embed(banner1=banner, local_ns=local_ns)


if __name__ == "__main__":
    main()
