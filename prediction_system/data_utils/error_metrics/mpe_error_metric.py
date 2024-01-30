import numpy as np

from prediction_system.data_utils.error_metrics.base_error_metric import ErrorMetric


class MPEMetric(ErrorMetric):
    EPSILON = 1e-10

    @staticmethod
    def _compute(real: np.array, prediction: np.array) -> float:
        return np.mean((real - prediction) / (real + MPEMetric.EPSILON)) * 100
