from abc import abstractmethod
from datetime import timedelta

import pandas as pd

from prediction_system.data_utils.results_data import ResultsData


class BaseTimeSeriesDataCorrector:
    """
    Uses feedback loop to correct the data

    Args:
        data: Data to correct
        training_data_period: Period of training data in days
        test_period: Period of test data in days
        test_data_delay: Delay of test data in days
    """
    def __init__(
        self,
        data: ResultsData,
        training_data_period: int = 28,
        test_period: int = 1,
        test_data_delay: int = 0
    ):
        self.data = data
        self.training_data_period = training_data_period
        self.test_period = test_period
        self.test_data_delay = test_data_delay

    @abstractmethod
    def _get_corrected_prediction(self) -> pd.Series:
        """
        :return: Corrected prediction
        """
        pass

    def correct(self) -> ResultsData:
        """
        Corrects the data

        Returns:
            Corrected data
        """
        corrected_data = self.data.to_dataframe().copy()
        corrected_data.prediction = self._get_corrected_prediction()
        return ResultsData.from_dataframe(corrected_data)
