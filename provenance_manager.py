"""
This module manages the provenance information for each tuple in a DataFrame.
"""

from typing import Any, Optional
import pandas as pd
from consts import PROVENANCE
from semiring import Semiring


class ProvenanceManager:
    """
    Manages the provenance information for each tuple in a DataFrame.

    Attributes:
        semiring (Semiring): The semiring instance used for provenance tokens.
    """

    def __init__(self, semiring: Semiring):
        self.semiring = semiring

    def assign_provenance(
        self,
        df: pd.DataFrame,
        table_name: str,
        provenance_col_name: Optional[str] = PROVENANCE,
    ) -> pd.DataFrame:
        """
        Assigns provenance values to each row in the DataFrame using the semiring.

        Args:
            df (pd.DataFrame): The DataFrame to assign provenance to.
            table_name (str): The name of the table.
            provenance_col_name (Optional[str]): The name of the provenance column to use.
                If the column exists in the DataFrame, it will be converted to the semiring format.
                If not provided or not found, new provenance tokens will be assigned.

        Returns:
            pd.DataFrame: The DataFrame with provenance assigned, with the column named PROVENANCE.
        """
        df = df.copy()

        if provenance_col_name in df.columns:
            if provenance_col_name != PROVENANCE:
                # Rename the existing provenance column to the PROVENANCE constant
                df.rename(columns={provenance_col_name: PROVENANCE}, inplace=True)

            # Convert the existing provenance column to the semiring format
            df[PROVENANCE] = df[PROVENANCE].apply(lambda x: self.semiring.convert_provenance(x))
        else:
            # Create new provenance tokens for each row
            provenance_tokens = []
            for idx in df.index:
                token = self.semiring.get_provenance_token(table_name=table_name, idx=idx)
                provenance_tokens.append(token)
            df[PROVENANCE] = provenance_tokens

        return df

    def get_provenance(self, idx: int, df: pd.DataFrame) -> Any:
        """
        Retrieves the provenance expression for a given tuple index.

        Args:
            idx (int): The index of the tuple.
            df (pd.DataFrame): The DataFrame containing the provenance.

        Returns:
            Any: The provenance expression for the specified tuple.
        """
        return df.at[idx, PROVENANCE]

    def update_provenance(self, idx: int, df: pd.DataFrame, expression: Any):
        """
        Updates the provenance expression for a given tuple index.

        Args:
            idx (int): The index of the tuple.
            df (pd.DataFrame): The DataFrame containing the provenance.
            expression (Any): The new provenance expression to assign.
        """
        df.at[idx, PROVENANCE] = expression
