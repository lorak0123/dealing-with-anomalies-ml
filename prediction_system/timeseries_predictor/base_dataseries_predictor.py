from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd
from sklearn import clone
from tqdm import tqdm

from prediction_system.data_utils.input_data import InputData
from prediction_system.data_utils.results_data import ResultsData


class BaseTimeSeriesPredictor:
    """
    Class to process prediction for time labeled data
    """
    def __init__(self,
                 input_data: InputData,
                 training_data_period: int = 28,
                 test_period: int = 1,
                 test_data_delay: int = 0,
                 max_workers: int = 1):
        """
        Args:
            input_data: Input data
            training_data_period: Period of training data in days
            test_period: Period of test data in days
            test_data_delay: Delay of test data in days
            max_workers: Number of workers to use for parallel processing
        """
        self.input_data = input_data
        self.training_data_period = timedelta(days=training_data_period)
        self.test_period = timedelta(days=test_period)
        self.test_data_delay = timedelta(days=test_data_delay)
        self.max_workers = max_workers

    def calculate_periods_count(self) -> int:
        """
        :return: Number of periods in the data
        """
        return int(
            (
                self.input_data.index.max()
                - self.input_data.index.min()
                - self.training_data_period
            ) / self.test_period)

    def split(self) -> (pd.DataFrame, pd.DataFrame):
        """
        :return: Generator of train and test data sets
        """
        indexes = self.input_data.index
        start = self.input_data.index.min() + self.training_data_period
        while start + self.test_period <= self.input_data.index.max() - self.test_data_delay:
            end = start + self.test_period
            train = self.input_data.loc((indexes >= start - self.training_data_period) & (indexes <= start))
            test = self.input_data.loc(
                (indexes >= start+self.test_data_delay) & (indexes < end+self.test_data_delay)
            )
            yield train, test
            start = end

    @staticmethod
    def run_in_runner(train: InputData, test: InputData, model) -> ResultsData:
        model.fit(train.data, train.real)
        return ResultsData.from_input_data(
            test,
            pd.Series(model.predict(test.data), index=test.index, name='prediction')
        )

    def run_prediction(self, model) -> ResultsData:
        """
        :param model: Model set to run predictions
        :return: PredictionResults object
        """
        results = []

        if self.max_workers > 1:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)

            with executor as executor:

                progress_bar = tqdm(total=self.calculate_periods_count())
                def callback(future):
                    results.append(future.result())
                    progress_bar.update()

                for train, test in self.split():
                    future = executor.submit(
                        self.run_in_runner,
                        train=train,
                        test=test,
                        model=clone(model)
                    )

                    future.add_done_callback(callback)

            progress_bar.colour = 'green'
            results = pd.concat([result.to_dataframe() for result in results])
        else:
            for train, test in tqdm(list(self.split())):
                model_copy = clone(model)
                model_copy.fit(train.data, train.real)
                results += ResultsData.from_input_data(
                    test,
                    pd.Series(
                        model_copy.predict(test.data),
                        index=test.index,
                        name='prediction'
                    )
                )

        return results
