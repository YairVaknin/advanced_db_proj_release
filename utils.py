"""
This module defines utility functions used throughout the project.
"""

import os

import pandas as pd
from query_processor import QueryProcessor


def load_custom_data(qp: QueryProcessor, csv_dir: str):
    """
    Loads CSV files from a custom directory into the QueryProcessor.

    Args:
        qp (QueryProcessor): The QueryProcessor instance to load data into.
        csv_dir (str): The directory path containing CSV files.

    The function reads all CSV files in the specified directory and loads them into the
    QueryProcessor with assigned provenance and probabilities.
    """
    if not os.path.isdir(csv_dir):
        print(f"Error: The provided path '{csv_dir}' is not a directory.")
        return

    # Iterate over each CSV file in the directory
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_dir, filename)
            table_name = os.path.splitext(filename)[
                0
            ]  # Use filename without extension as table name

            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(filepath)

            # Load the DataFrame into the QueryProcessor
            qp.load_table(df=df, table_name=table_name)

            print(
                f"Loaded {filename} into table '{table_name}' with assigned provenance and"
                " probabilities."
            )
    print(f"All CSV files from '{csv_dir}' have been loaded into the QueryProcessor.")
