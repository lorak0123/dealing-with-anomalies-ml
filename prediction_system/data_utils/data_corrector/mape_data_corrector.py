import pandas as pd

from prediction_system.data_utils.data_corrector.base_data_corrector import BaseTimeSeriesDataCorrector


class MAPEDataCorrector(BaseTimeSeriesDataCorrector):
    def _get_corrected_prediction(self) -> pd.Series:
        groups = self.data.index.map(lambda x: x.date())
        grouped_data = (self.data.real - self.data.prediction).groupby(groups).mean()

        return
