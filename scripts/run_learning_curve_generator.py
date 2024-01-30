import logging
from pathlib import Path
import click as click
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from prediction_system.data_utils.error_metrics import get_error_metric_by_name
from prediction_system.data_utils.error_metrics.mae_error_metric import MAEErrorMetric
from prediction_system.data_utils.helpers import interpolate_data
from prediction_system.data_utils.results_data import ResultsData

from data import DATA_DIR
from prediction_system.data_utils.path_manager import prepare_directory
from utils.exception_logger_decorator import exception_logger

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    '--input_path',
    type=click.Path(),
    default=DATA_DIR / 'evaluation/models_evaluation',
    help='Path to the data directory'
)
@click.option(
    '--show_plot',
    default=True,
    help='Show plot'
)
@click.option(
    '--interpolate',
    default=True,
    help='Interpolate data'
)
@click.option(
    '--error_metric',
    default='MAE',
    help='Error metric to use for evaluation. Must be defined in prediction_system/data_utils/error_metrics/__init__.py'
)
@exception_logger
def generate_learning_curves(input_path: Path, show_plot: bool, interpolate: bool, error_metric: str) -> None:
    stats = {}
    error_metric_class = get_error_metric_by_name(error_metric)
    progress_bar = tqdm(list(input_path.glob('*.csv')), desc='Reading files')
    for file in progress_bar:
        results = ResultsData.from_csv(file)

        model, data_size = file.stem.split('_')

        if model not in stats:
            stats[model] = pd.DataFrame(
                {'err': [error_metric_class.compute(results)]},
                index=[int(data_size)]
            )

        else:
            stats[model].loc[int(data_size)] = [error_metric_class.compute(results)]

    progress_bar.colour = 'green'

    fig, ax = plt.subplots(figsize=(20, 10))

    for model, data in stats.items():
        data.sort_index(inplace=True)
        if interpolate:
            ax.plot(*interpolate_data(data.index, data.err, precision=1), label=model)
        else:
            ax.plot(data.index, data.mae, label=model)

    ax.set_title('Model evaluation')
    ax.set_xlabel('Data size')
    ax.set_ylabel(error_metric)
    ax.grid()
    ax.legend()
    plt.savefig(prepare_directory(input_path) / 'model_evaluation.png')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    generate_learning_curves()
