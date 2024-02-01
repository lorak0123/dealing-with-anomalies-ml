import numpy as np

from prediction_system.data_utils.error_metrics.base_error_metric import ErrorMetric


class SMAPEMetric(ErrorMetric):
    @staticmethod
    def _compute(real: np.array, prediction: np.array) -> float:
        return np.mean(np.abs(real - prediction) / (np.abs(real) + np.abs(prediction))) * 200
