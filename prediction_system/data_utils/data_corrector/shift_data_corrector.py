import pandas as pd

from prediction_system.data_utils.data_corrector.base_data_corrector import BaseTimeSeriesDataCorrector


class ShiftDataCorrector(BaseTimeSeriesDataCorrector):
    def _get_corrected_prediction(self) -> pd.Series:
        return self.data.prediction + (self.data.real - self.data.prediction).shift(
            (self.test_data_delay + self.test_period) * 24
        ).fillna(0)
