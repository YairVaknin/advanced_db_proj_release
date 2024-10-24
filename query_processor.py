"""
This module defines the QueryProcessor class, which handles database operations and provenance tracking.
"""

import functools
import pickle
import random
from typing import Dict, List, Literal, Optional, Union
import pandas as pd
from sympy import Symbol  # type: ignore

from consts import ADD, BAG, MUL, PROBABILITY, PROVENANCE, SET
from expression_handler import ExpressionHandler
from probability_calc import compute_probability
from provenance_manager import ProvenanceManager
from semiring import BooleanPolynomialSemiring, PolynomialSemiring, Semiring


class QueryProcessor:
    """
    Query Processor class to handle database operations and provenance tracking.

    Attributes:
        semiring (Semiring): The semiring instance used for provenance calculations.
        pm (ProvenanceManager): Manages provenance assignments.
        eh (ExpressionHandler): Handles provenance expression combinations.
        tables (Dict[str, pd.DataFrame]): Mapping of table names to DataFrames with provenance.
        semantics (str): Semantics of the database ('set' or 'bag').
        provenance_probability_map (Dict[Symbol, float]): Mapping of provenance symbols to probabilities.
        compute_probability (Callable): Function to compute probabilities of provenance expressions.
    """

    def __init__(
        self,
        semantics: str = SET,
        semiring: Semiring = PolynomialSemiring(),
    ) -> None:
        self.semiring: Semiring = semiring
        self.pm: ProvenanceManager = ProvenanceManager(semiring=self.semiring)
        self.eh: ExpressionHandler = ExpressionHandler(semiring=self.semiring)
        self.tables: Dict[str, pd.DataFrame] = (
            {}
        )  # Mapping of table names to DataFrames with provenance
        self.semantics: str = semantics  # Semantics of the database - 'set' or 'bag'
        # Mapping of provenance symbols in the database to their probabilities
        self.provenance_probability_map: Dict[Symbol, float] = {}
        # Partial function to compute probabilities based on the semiring
        self.compute_probability = functools.partial(
            compute_probability,
            semiring=self.semiring,
            probabilities=self.provenance_probability_map,
        )
        # Update the wrapper to include the original function's metadata
        functools.update_wrapper(self.compute_probability, compute_probability)

    def save_qp(self, filepath: str) -> None:
        """
        Saves the current QueryProcessor instance to a file.

        Args:
            filepath (str): The path where the QueryProcessor instance will be saved.
        """
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"QueryProcessor instance saved to {filepath}.")

    @staticmethod
    def load_qp(filepath: str) -> "QueryProcessor":
        """
        Loads a QueryProcessor instance from a file.

        Args:
            filepath (str): The path from where the QueryProcessor instance will be loaded.

        Returns:
            QueryProcessor: The loaded QueryProcessor instance.
        """
        with open(filepath, "rb") as file:
            qp_instance = pickle.load(file)
        print(f"QueryProcessor instance loaded from {filepath}.")
        return qp_instance

    def get_available_tables(self, ret_string=False) -> Optional[str]:
        """
        Returns a string listing the names of the available tables.

        Returns:
            str: A formatted string of available table names.
        """
        keys = [""] + list(self.tables.keys())
        str = "\n * ".join(keys)
        if ret_string:
            return str
        return print(f"\nAvailable tables:{str}")

    def insert_table(self, table_name: str, df: pd.DataFrame) -> bool:
        """
        Inserts a DataFrame into the database and assigns it to the specified table name.
        Should be used for results of queries in most cases.

        Args:
            table_name (str): The name of the table.
            df (pd.DataFrame): The DataFrame to insert.

        Returns:
            bool: True if the table was inserted successfully, False otherwise.
        """
        if table_name in self.tables:
            print(f"Table '{table_name}' already exists. Please use replace_table() to update it.")
            return False
        self.tables[table_name] = df
        return True

    def replace_table(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Replaces an existing table in the database with a new DataFrame.

        Args:
            table_name (str): The name of the table to replace.
            df (pd.DataFrame): The new DataFrame to associate with the table name.
        """
        if table_name not in self.tables:
            print(f"Table '{table_name}' does not exist. Please use insert_table() to create it.")
            return
        self.tables[table_name] = df

    def load_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        provenance_col_name: Optional[str] = PROVENANCE,
        probability_col_name: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Loads a DataFrame into the database and assigns or uses existing provenance.

        Args:
            df (pd.DataFrame): The DataFrame to load.
            table_name (str): The name of the table.
            provenance_col_name (Optional[str]): The name of the provenance column in the DataFrame.
                If provided and exists, the DataFrame is expected to have this column.
                If not provided, a new provenance column is assigned using the semiring.
            probability_col_name (Optional[str]): The name of the probability column in the DataFrame.
                If provided, probabilities are taken from this column (nulls are treated as 1.0).
                If not provided, random probabilities are generated and added to the DataFrame.

        Returns:
            Optional[pd.DataFrame]: The DataFrame with provenance (and probabilities if applicable),
            or None if insertion failed.
        """
        # Assign or use existing provenance
        df_with_provenance = self.pm.assign_provenance(
            df.copy(), table_name=table_name, provenance_col_name=provenance_col_name
        )

        # Register the table with the database connection
        if not self.insert_table(table_name=table_name, df=df_with_provenance):
            # Table already exists. Insert failed.
            return None

        # Process probabilities if the semiring supports it
        if isinstance(self.semiring, (PolynomialSemiring, BooleanPolynomialSemiring)):
            self.process_probabilities(
                df=df_with_provenance,
                table_name=table_name,
                probability_col_name=probability_col_name,
            )
        return df_with_provenance

    def process_probabilities(
        self,
        df: pd.DataFrame,
        table_name: str,
        probability_col_name: Optional[str],
    ):
        """
        Processes the probability mappings for the given table.

        Args:
            df (pd.DataFrame): The DataFrame containing the data and provenance.
            table_name (str): The name of the table.
            probability_col_name (Optional[str]): The name of the probability column in the DataFrame.
                If provided, probabilities are taken from this column.
                If not provided, random probabilities are generated and added to the DataFrame.

        Raises:
            AssertionError: If probabilities are invalid or if the semiring is not PolynomialSemiring.
        """
        provenance_values = df[PROVENANCE]
        probabilities = None

        if probability_col_name and probability_col_name in df.columns:
            # Use provided probabilities
            probabilities = df[probability_col_name].fillna(1.0)
            # Assert that all probabilities are valid numbers between 0 and 1
            assert probabilities.between(0, 1).all(), "Probabilities must be between 0 and 1."
            if probability_col_name != PROBABILITY:
                # Rename the column to PROBABILITY if it's different
                df.rename(columns={probability_col_name: PROBABILITY}, inplace=True)
        else:
            # Generate random probabilities and add them to the DataFrame
            probabilities = pd.Series([random.uniform(0, 1) for _ in df.index], index=df.index)
            df[PROBABILITY] = probabilities
            probability_col_name = PROBABILITY

        # Create the mapping between provenance symbols and probabilities
        provenance_to_probability = dict(zip(provenance_values, probabilities))

        # Add the mapping to the global dictionary
        self.provenance_probability_map.update(provenance_to_probability)

        # Drop the probability column; computations will use the mapping
        df.drop(columns=[PROBABILITY], inplace=True)

    def _get_table_df(
        self, table: Union[str, pd.DataFrame], arg_name: str = "table"
    ) -> pd.DataFrame:
        """
        Helper method to retrieve a DataFrame from a table name or validate a DataFrame.

        Args:
            table (Union[str, pd.DataFrame]): The table to retrieve or validate.
            arg_name (str): The argument name, used for error messages.

        Returns:
            pd.DataFrame: The retrieved or validated DataFrame.

        Raises:
            ValueError: If the table name is not found in self.tables.
            TypeError: If the provided table is neither a string nor a DataFrame.
        """
        if isinstance(table, str):
            if table not in self.tables:
                raise ValueError(f"Table '{table}' not found in QueryProcessor.")
            return self.tables[table]
        elif isinstance(table, pd.DataFrame):
            return table
        else:
            raise TypeError(f"{arg_name} must be a string (table name) or a pandas DataFrame.")

    def _handle_semantics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to handle set or bag semantics based on the QueryProcessor instance.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The processed DataFrame based on the semantics.
        """
        if self.semantics == SET:
            # Eliminate duplicates and sum provenance expressions
            data_columns: List[str] = [col for col in df.columns if col != PROVENANCE]
            grouped_df: pd.DataFrame = df.groupby(data_columns, as_index=False).agg(
                {
                    PROVENANCE: lambda x: self.eh.combine_provenance(
                        expressions=list(x), operation=ADD
                    )
                }
            )
            return grouped_df.reset_index(drop=True)
        else:
            assert self.semantics == BAG, "Semantics must be 'set' or 'bag'."
            # Bag semantics: retain duplicates
            return df.reset_index(drop=True)

    def select(
        self,
        table: Union[str, pd.DataFrame],
        condition: str,
    ) -> pd.DataFrame:
        """
        Performs a selection operation on a table based on a condition.

        Args:
            table (Union[str, pd.DataFrame]): The table to perform selection on.
                Can be a table name or a DataFrame.
            condition (str): The condition string to use for filtering the DataFrame.

        Returns:
            pd.DataFrame: The resulting DataFrame after applying the selection.
        """
        df = self._get_table_df(table, arg_name="table")

        selected_df: pd.DataFrame = df.query(condition).copy()

        return self._handle_semantics(selected_df)

    def project(
        self,
        table: Union[str, pd.DataFrame],
        columns: List[str],
    ) -> pd.DataFrame:
        """
        Performs a projection operation to select specific columns from a table.

        Args:
            table (Union[str, pd.DataFrame]): The table to perform projection on.
                Can be a table name or a DataFrame.
            columns (List[str]): The list of columns to project.

        Returns:
            pd.DataFrame: The resulting DataFrame after applying the projection.
        """
        df = self._get_table_df(table, arg_name="table")

        projected_df: pd.DataFrame = df[columns + [PROVENANCE]].copy()

        return self._handle_semantics(projected_df)

    def join(
        self,
        left: Union[str, pd.DataFrame],
        right: Union[str, pd.DataFrame],
        on: Optional[List[str]] = None,
        how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
    ) -> pd.DataFrame:
        """
        Performs a join operation between two tables and updates provenance.

        This method supports various types of joins, including Cartesian product,
        theta join, equi-join, and natural join, by utilizing the `how` and `on` parameters.

        Args:
            left (Union[str, pd.DataFrame]): The left table for the join.
                Can be a table name or a DataFrame.
            right (Union[str, pd.DataFrame]): The right table for the join.
                Can be a table name or a DataFrame.
            on (Optional[List[str]]): Column names to join on.
                - For an equi-join or natural join, specify the columns to join on.
                - For a natural join, if `on` is None, the join will be performed on all common columns.
                - For a Cartesian product or theta join, set `on=None`.
            how (str): Type of join to perform. Options are:
                - 'inner' (default): Returns records that have matching values in both tables.
                - 'left': Returns all records from the left table, and the matched records from the right table.
                - 'right': Returns all records from the right table, and the matched records from the left table.
                - 'outer': Returns all records when there is a match in either left or right table.
                - 'cross': Performs a Cartesian product (all combinations of rows from both tables).

        Returns:
            pd.DataFrame: The resulting DataFrame after applying the join.

        Usage Examples:
            - **Cartesian Product**:
                To perform a Cartesian product of two tables `R` and `S`:
                ```python
                result = qp.join(left='R', right='S', how='cross')
                ```
            - **Theta Join**:
                To perform a theta join with a condition `theta_condition`:
                ```python
                # First, perform a Cartesian product
                cartesian_df = qp.join(left='R', right='S', how='cross')
                # Then, apply the selection with the theta condition
                result = qp.select(table=cartesian_df, condition=theta_condition)
                ```
            - **Equi-Join**:
                To perform an equi-join on columns `a` and `b`:
                ```python
                result = qp.join(left='R', right='S', on=['a', 'b'])
                ```
            - **Natural Join**:
                To perform a natural join on all common columns:
                ```python
                result = qp.join(left='R', right='S')
                ```
                This will automatically join on all columns with the same names in both tables.

        Notes:
            - The provenance expressions of joined tuples are combined using semiring multiplication
            to reflect the dependency on both input tuples.
            - The method respects the set or bag semantics defined in the `QueryProcessor` instance,
            affecting how duplicates are handled in the result.
        """
        left_df = self._get_table_df(left, arg_name="left")
        right_df = self._get_table_df(right, arg_name="right")

        joined_df: pd.DataFrame = left_df.merge(
            right=right_df, on=on, how=how, suffixes=("_left", "_right")
        )

        # Combine provenance expressions
        joined_df[PROVENANCE] = joined_df.apply(
            lambda row: self.eh.combine_provenance(
                expressions=[row["provenance_left"], row["provenance_right"]], operation=MUL
            ),
            axis=1,
        )
        # Drop the old provenance columns
        joined_df.drop(columns=["provenance_left", "provenance_right"], inplace=True)

        return self._handle_semantics(joined_df)

    def union(
        self,
        table1: Union[str, pd.DataFrame],
        table2: Union[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Performs a union operation between two tables and updates provenance.

        In set semantics, duplicates are removed, and provenance expressions are summed.
        In bag semantics, duplicates are retained, and provenance expressions are not summed.

        Args:
            table1 (Union[str, pd.DataFrame]): The first table for the union.
                Can be a table name or a DataFrame.
            table2 (Union[str, pd.DataFrame]): The second table for the union.
                Can be a table name or a DataFrame.

        Returns:
            pd.DataFrame: The resulting DataFrame after applying the union.
        """
        df1 = self._get_table_df(table1, arg_name="table1")
        df2 = self._get_table_df(table2, arg_name="table2")

        # Concatenate the two DataFrames
        union_df: pd.DataFrame = pd.concat([df1, df2], ignore_index=True)

        return self._handle_semantics(union_df)

    def intersection(
        self,
        table1: Union[str, pd.DataFrame],
        table2: Union[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Performs an intersection of two tables and updates provenance.

        Only rows that are present in both tables are included.

        Args:
            table1 (Union[str, pd.DataFrame]): The first table for the intersection.
                Can be a table name or a DataFrame.
            table2 (Union[str, pd.DataFrame]): The second table for the intersection.
                Can be a table name or a DataFrame.

        Returns:
            pd.DataFrame: The resulting DataFrame after applying the intersection.
        """
        df1 = self._get_table_df(table1, arg_name="table1")
        df2 = self._get_table_df(table2, arg_name="table2")

        # Perform an inner join on all columns except provenance
        data_columns: List[str] = [col for col in df1.columns if col != PROVENANCE]

        # Inner join on all columns except provenance
        intersected_df: pd.DataFrame = df1.merge(
            df2, on=data_columns, how="inner", suffixes=("_left", "_right")
        )

        # Combine provenance expressions from both DataFrames for rows that intersect
        intersected_df[PROVENANCE] = intersected_df.apply(
            lambda row: self.eh.combine_provenance(
                expressions=[row["provenance_left"], row["provenance_right"]],
                operation=MUL,
            ),
            axis=1,
        )

        # Drop old provenance columns
        intersected_df.drop(columns=["provenance_left", "provenance_right"], inplace=True)

        return self._handle_semantics(intersected_df)

    def difference(
        self,
        table1: Union[str, pd.DataFrame],
        table2: Union[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Performs a difference between two tables and updates provenance.

        Returns rows that are in table1 but not in table2.

        Args:
            table1 (Union[str, pd.DataFrame]): The first table (minuend) for the difference.
                Can be a table name or a DataFrame.
            table2 (Union[str, pd.DataFrame]): The second table (subtrahend) for the difference.
                Can be a table name or a DataFrame.

        Returns:
            pd.DataFrame: The resulting DataFrame after applying the difference.
        """
        df1 = self._get_table_df(table1, arg_name="table1")
        df2 = self._get_table_df(table2, arg_name="table2")

        # Perform a left join on all columns except provenance
        data_columns: List[str] = [col for col in df1.columns if col != PROVENANCE]

        # Left join on all columns except provenance
        difference_df: pd.DataFrame = df1.merge(
            df2, on=data_columns, how="left", suffixes=("_left", "_right")
        )

        # Filter rows where the right side columns are NaN (i.e., no match)
        difference_df = difference_df[difference_df["provenance_right"].isna()].copy()

        # Keep only the provenance from the left side
        difference_df[PROVENANCE] = difference_df["provenance_left"]

        # Drop unnecessary columns
        difference_df.drop(columns=["provenance_left", "provenance_right"], inplace=True)

        return self._handle_semantics(difference_df)

    def rename(
        self,
        table: Union[str, pd.DataFrame],
        columns: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Renames columns in the given table.

        Args:
            table (Union[str, pd.DataFrame]): The table to perform rename on.
                Can be a table name or a DataFrame.
            columns (Dict[str, str]): A dictionary mapping old column names to new column names.

        Returns:
            pd.DataFrame: The resulting DataFrame after renaming the columns.
        """
        df = self._get_table_df(table, arg_name="table")

        # We don't allow renaming the provenance column.
        if PROVENANCE in columns.values() or PROVENANCE in columns.keys():
            raise ValueError(f"Cannot rename from/to '{PROVENANCE}' column.")

        # Check if all old column names exist in the DataFrame
        missing_columns = [col for col in columns.keys() if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in the table.")

        # Check that no duplicate column names could be created
        new_column_names = [col for col in columns.values()] + [
            col for col in df.columns if col not in columns.keys()
        ]
        if len(new_column_names) != len(set(new_column_names)):
            raise ValueError(
                "Duplicate column names would be created after renaming. Please make sure new"
                " column names are unique."
            )

        # Rename the columns
        renamed_df = df.rename(columns=columns)

        return renamed_df.reset_index(drop=True)
