import pandas as pd

from prediction_system.data_utils.input_data import InputData


class ResultsData(InputData):
    """
    Class that represents the results data for the prediction system.
    """

    def __init__(self, data: pd.DataFrame):
        if 'prediction' not in data.columns:
            raise ValueError('The data must have a `prediction` column.')
        self._validate_data(data)
        self._prediction: pd.Series = data['prediction']
        super().__init__(data.drop(columns=['prediction']))

    @staticmethod
    def from_input_data(input_data: InputData, prediction: pd.Series) -> 'ResultsData':
        """
        Creates a new ResultsData object from the given input data and prediction.

        Args:
            input_data: The input data.
            prediction: The prediction.

        Returns:
            A new ResultsData object.
        """
        return ResultsData(pd.concat([input_data.to_dataframe(), prediction], axis=1))

    @property
    def prediction(self) -> pd.Series:
        """
        Returns a copy of the prediction data
        """
        return self._prediction.copy()

    @classmethod
    def get_empty(cls) -> 'ResultsData':
        return cls(pd.DataFrame(columns=['real', 'prediction']))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([self._data, self._real, self._prediction], axis=1)
