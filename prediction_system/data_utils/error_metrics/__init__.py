from prediction_system.data_utils.error_metrics.base_error_metric import ErrorMetric
from prediction_system.data_utils.error_metrics.mae_error_metric import MAEErrorMetric

MAE = MAEErrorMetric()


def get_error_metric_by_name(name: str) -> ErrorMetric:
    """
    Returns the error metric with the given name.

    Args:
        name: The name of the error metric.

    Returns:
        The error metric with the given name.
    """
    try:
        return globals()[name]
    except KeyError:
        raise ValueError(f'Error metric {name} not found. Please, check if it is defined in '
                         'prediction_system/data_utils/error_metrics/__init__.py')
