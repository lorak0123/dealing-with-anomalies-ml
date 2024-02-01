import os

from prediction_system.data_utils.error_metrics.maape_metric import MAAPEMetric

from prediction_system.data_utils.error_metrics.mae_metric import MAEMetric

from prediction_system.data_utils.error_metrics.base_error_metric import ErrorMetric
from prediction_system.data_utils.error_metrics.mape_metric import MAPEMetric
from prediction_system.data_utils.error_metrics.me_metric import MEMetric
from prediction_system.data_utils.error_metrics.mpe_metric import MPEMetric
from prediction_system.data_utils.error_metrics.mse_metric import MSEMetric
from prediction_system.data_utils.error_metrics.nmae_metric import NMAEMetric
from prediction_system.data_utils.error_metrics.rmse_metric import RMSEMetric
from prediction_system.data_utils.error_metrics.smape_metric import SMAPEMetric

DEFAULT_DATA_UNIT = os.environ.get('DEFAULT_DATA_UNIT', None)

ME = MEMetric(name='ME', full_name='Mean Error', unit=DEFAULT_DATA_UNIT)
MAE = MAEMetric(name='MAE', full_name='Mean Absolute Error', unit=DEFAULT_DATA_UNIT)
MPE = MPEMetric(name='MPE', unit='%', full_name='Mean Percentage Error')
MAPE = MAPEMetric(name='MAPE', unit='%', full_name='Mean Absolute Percentage Error')
MSE = MSEMetric(name='MSE', full_name='Mean Squared Error', unit=f"{DEFAULT_DATA_UNIT}^2")
RMSE = RMSEMetric(name='RMSE', full_name='Root Mean Squared Error', unit=DEFAULT_DATA_UNIT)
MAAPE = MAAPEMetric(name='MAAPE', unit='%', full_name='Mean Arctangent Absolute Percentage Error')
NMAE = NMAEMetric(name='NMAE', full_name='Normalized Mean Absolute Error', unit='%')
SMAPE = SMAPEMetric(name='SMAPE', unit='%', full_name='Symmetric Mean Absolute Percentage Error')


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
