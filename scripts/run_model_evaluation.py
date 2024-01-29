import logging
from pathlib import Path
import click as click

from data import DATA_DIR
from prediction_system.config import get_model_by_name
from prediction_system.data_utils.input_data import InputData
from prediction_system.data_utils.path_manager import prepare_directory
from prediction_system.timeseries_predictor.base_dataseries_predictor import BaseTimeSeriesPredictor
from utils.time_stats_utils import time_stats_decorator

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    '--data_path',
    type=click.Path(),
    default=DATA_DIR / 'data.csv',
    help='Path to the data file'
)
@click.option(
    '--output_path',
    default=DATA_DIR / 'evaluation/models_evaluation',
    type=click.Path(),
    help='Path to the output directory'
)
@click.option(
    '--step',
    default=[1, 4, 8, 16, 20, 24, 30, 40, 60, 100],
    multiple=True,
    help='Steps to use for learning curves'
)
@click.option(
    '--model',
    default='ALL_MODELS',
    help='Model to use for prediction. Must be defined in prediction_system/config.py'
)
@click.option(
    '--n_jobs',
    default=20,
    help='Number of parallel jobs to run'
)
@time_stats_decorator(category='learning_curves')
def model_evaluation(data_path: Path, output_path: Path, step: list, model: str, n_jobs: int) -> None:
    """
    Generates learning curves for the given data file and saves them to the given output directory.

    Args:
        data_path: Path to the data file
        output_path: Path to the output directory
        step: Steps to use for learning curves
        model: Model to use for prediction. Must be defined in prediction_system/config.py
        n_jobs: Number of parallel jobs to run
    """
    input_data = InputData.from_csv(data_path)

    models = get_model_by_name(model)

    models = [models] if isinstance(models, str) else models

    for model in models:
        logging.info(f'Generating learning curves for {model.__class__.__name__}')
        for s in step:
            predictor = BaseTimeSeriesPredictor(
                input_data=input_data,
                training_data_period=s,
                test_period=1,
                test_data_delay=0,
                max_workers=n_jobs,
            )
            time_stats_decorator()(predictor.run_prediction)(model).to_csv(
                prepare_directory(output_path) / f'{model.__class__.__name__}_{s}.csv'
            )


if __name__ == '__main__':
    model_evaluation()
