from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd


class DataFrameInterface(ABC):
    """
    Interface for classes that can be converted to and from pandas DataFrames.
    """
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the object to a pandas DataFrame.

        Returns:
            The pandas DataFrame representation of the object.
        """
        ...

    @abstractmethod
    def from_dataframe(self, data: pd.DataFrame) -> Self:
        """
        Creates an object from a pandas DataFrame.

        Args:
            data: The pandas DataFrame to convert.

        Returns:
            The object representation of the pandas DataFrame.
        """
        ...

    @classmethod
    def from_csv(cls, path: Path) -> Self:
        """
        Creates an object from a CSV file.

        Args:
            path: The path to the CSV file.

        Returns:
            The object representation of the CSV file.
        """
        return cls.from_dataframe(data=pd.read_csv(path, index_col=0))

    def to_csv(self, path: Path) -> None:
        """
        Saves the object to a CSV file.

        Args:
            path: The path to the CSV file.
        """
        self.to_dataframe().to_csv(path)

    def loc(self, indexes: pd.DatetimeIndex) -> Self:
        """
        Returns a copy of the data with the specified indexes
        """
        return self.from_dataframe(self.to_dataframe().loc[indexes])

    def __add__(self, other) -> Self:
        if (this := self.to_dataframe()).empty:
            return other
        if (other := other.to_dataframe()).empty:
            return self

        return self.from_dataframe(
            pd.concat([this, other]).sort_index()
        )
