from prediction_system.data_utils.error_metrics.maape_error_metric import MAAPEMetric

from prediction_system.data_utils.error_metrics.mae_error_metric import MAEMetric

from prediction_system.data_utils.error_metrics.base_error_metric import ErrorMetric
from prediction_system.data_utils.error_metrics.mape_error_metric import MAPEMetric
from prediction_system.data_utils.error_metrics.me_error_metric import MEMetric
from prediction_system.data_utils.error_metrics.mpe_error_metric import MPEMetric
from prediction_system.data_utils.error_metrics.rmse_error_metric import RMSEMetric

MAE = MAEMetric(name='MAE', full_name='Mean Absolute Error')
ME = MEMetric(name='ME', full_name='Mean Error')
MPE = MPEMetric(name='MPE', unit='%', full_name='Mean Percentage Error')
MAPE = MAPEMetric(name='MAPE', unit='%', full_name='Mean Absolute Percentage Error')
MAAPE = MAAPEMetric(name='MAAPE', unit='%', full_name='Mean Arctangent Absolute Percentage Error')
RMSE = RMSEMetric(name='RMSE', full_name='Root Mean Squared Error')


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
