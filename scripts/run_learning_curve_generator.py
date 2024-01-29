import logging
from pathlib import Path
import click as click
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from prediction_system.data_utils.helpers import interpolate_data
from prediction_system.data_utils.results_data import ResultsData

from data import DATA_DIR
from prediction_system.config import get_model_by_name
from prediction_system.data_utils.input_data import InputData
from prediction_system.data_utils.path_manager import prepare_directory
from prediction_system.timeseries_predictor.base_dataseries_predictor import BaseTimeSeriesPredictor
from utils.time_stats_utils import time_stats_decorator

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
def generate_learning_curves(input_path: Path, show_plot: bool, interpolate: bool) -> None:
    stats = {}
    progress_bar = tqdm(list(input_path.glob('*.csv')), desc='Reading files')
    for file in progress_bar:
        results = ResultsData.from_csv(file)

        model, data_size = file.stem.split('_')

        if model not in stats:
            stats[model] = pd.DataFrame(
                {'mae': [(results.real - results.prediction).abs().mean()]},
                index=[int(data_size)]
            )

        else:
            stats[model].loc[int(data_size)] = [(results.real - results.prediction).abs().mean()]

    progress_bar.colour = 'green'

    fig, ax = plt.subplots(figsize=(20, 10))

    for model, data in stats.items():
        data.sort_index(inplace=True)
        if interpolate:
            ax.plot(*interpolate_data(data.index, data.mae, precision=1), label=model)
        else:
            ax.plot(data.index, data.mae, label=model)

    ax.set_title('Model evaluation')
    ax.set_xlabel('Data size')
    ax.set_ylabel('MAE')
    ax.grid()
    ax.legend()
    plt.savefig(prepare_directory(input_path) / 'model_evaluation.png')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    generate_learning_curves()
