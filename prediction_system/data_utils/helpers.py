import numpy as np
from scipy.interpolate import interp1d


def interpolate_data(
        x: list[float],
        y: list[float],
        precision: float = 0.1,
        kind: str = 'cubic'
) -> (np.array, np.array):
    """
    Interpolates the data to a given precision.

    Args:
        x: Range of the data.
        y: Data to interpolate.
        precision: Precision of the interpolation.
        kind: Kind of interpolation


    Returns:
        Tuple with the interpolated data.
    """

    f = interp1d(x, y, kind=kind)
    x_new = np.arange(np.array(x).min(), np.array(x).max(), precision)
    return x_new, f(x_new)
