import numpy as np

from prediction_system.data_utils.error_metrics.base_error_metric import ErrorMetric


class RMSEMetric(ErrorMetric):
    @staticmethod
    def _compute(real: np.array, prediction: np.array) -> float:
        return np.sqrt(np.square(real - prediction).mean())
