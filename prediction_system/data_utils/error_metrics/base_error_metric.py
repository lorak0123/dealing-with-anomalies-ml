from abc import abstractmethod

import numpy as np
import pandas as pd

from prediction_system.data_utils.results_data import ResultsData


class ErrorMetric:
    def __init__(
            self,
            name: str = None,
            full_name: str = None,
            unit: str = None,
    ):
        self._name = name
        self._full_name = full_name
        self._unit = unit

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @property
    def unit(self) -> str:
        return self._unit or ''

    @property
    def full_name(self) -> str:
        return self._full_name or self.name

    @staticmethod
    @abstractmethod
    def _compute(real: np.array, prediction: np.array) -> float:
        pass

    @classmethod
    def compute(cls, prediction_results: ResultsData, aggregation_function: callable = None) -> float | pd.Series:
        if aggregation_function is None:
            return cls._compute(prediction_results.real.values, prediction_results.prediction.values)
        else:
            return prediction_results.to_dataframe().groupby(aggregation_function).apply(
                lambda x: cls._compute(x.real.values, x.prediction.values)
            )
