from abc import abstractstaticmethod

import numpy as np
import pandas as pd

from prediction_system.data_utils.results_data import ResultsData


class ErrorMetric:
    @abstractstaticmethod
    def _compute(real: np.array, prediction: np.array) -> float:
        pass

    @classmethod
    def compute(cls, prediction_results: ResultsData, grouping_function: callable = None) -> float | pd.Series:
        if grouping_function is None:
            return cls._compute(prediction_results.real.values, prediction_results.prediction.values)
        else:
            return prediction_results.to_dataframe().groupby(grouping_function).apply(
                lambda x: cls._compute(x.real.values, x.prediction.values)
            )
