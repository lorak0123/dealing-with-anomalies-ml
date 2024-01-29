from typing import Self

import pandas as pd

from prediction_system.data_utils.dataframe_interface import DataFrameInterface


class InputData(DataFrameInterface):
    """
    Class that represents the input data for the prediction system.
    """
    def __init__(self, data: pd.DataFrame):
        self._validate_data(data)

        self._data: pd.DataFrame = data.drop(columns=['real'])
        self._real: pd.Series = data['real']

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([self._data, self._real], axis=1)

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        """
        Validates the data

        Args:
            data: The data to validate.

        Raises:
            ValueError: If the data is invalid.
        """
        if data.index.dtype != 'datetime64[ns]':
            try:
                data.index = pd.to_datetime(data.index)
            except ValueError:
                raise ValueError('The index must be a datetime64[ns] or convertible to it.')

        if 'real' not in data.columns:
            raise ValueError('The data must have a `real` column.')

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns a copy of the data
        """
        return self._data.copy()

    @property
    def real(self) -> pd.Series:
        """
        Returns a copy of the real data
        """
        return self._real.copy()

    @property
    def index(self) -> pd.DatetimeIndex:
        """
        Returns a copy of the index
        """
        return self._data.index.copy()

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> Self:
        return cls(data)

    @classmethod
    def get_empty(cls) -> 'InputData':
        return cls(pd.DataFrame(columns=['real']))
