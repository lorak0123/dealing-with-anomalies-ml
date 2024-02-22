import logging
from datetime import timedelta
from pathlib import Path
import click as click
from matplotlib import pyplot as plt
from prediction_system.data_utils.path_manager import prepare_directory

from prediction_system.data_utils.error_metrics import get_error_metric_by_name
from prediction_system.data_utils.results_data import ResultsData

from data import DATA_DIR
from utils.exception_logger_decorator import exception_logger

logging.basicConfig(level=logging.INFO)


AGGREGATION_FUNCTIONS = {
    'DAILY': lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0),
    'WEEKLY': lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=x.weekday()),
    'MONTHLY': lambda x: x.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
    'YEARLY': lambda x: x.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
}


def get_grouping_function(aggregation: str) -> callable:
    if aggregation not in AGGREGATION_FUNCTIONS:
        raise ValueError(
            f'Grouping {aggregation} not supported.\n'
            f'Try one of {list(AGGREGATION_FUNCTIONS.keys())}'
        )
    return AGGREGATION_FUNCTIONS[aggregation]


@click.command()
@click.option(
    '--input_file',
    type=click.Path(),
    help='Path to the results data file'
)
@click.option(
    '--output_path',
    type=click.Path(),
    default=DATA_DIR / 'evaluation/models_evaluation',
    help='Path to the output data directory'
)
@click.option(
    '--aggregation',
    default='DAILY',
    help='Aggregation function'
)
@click.option(
    '--show_plot',
    default=False,
    help='Show plot'
)
@click.option(
    '--error_metric',
    default='MAE',
    help='Error metric to use for evaluation. Must be defined in prediction_system/data_utils/error_metrics/__init__.py'
)
@exception_logger
def results_error_analytics(
    input_file: Path,
    output_path: Path,
    aggregation: str,
    show_plot: bool,
    error_metric: str
) -> None:
    results_data = ResultsData.from_csv(input_file)
    error_metric_class = get_error_metric_by_name(error_metric)
    aggregation_function = get_grouping_function(aggregation)

    error_data = error_metric_class.compute(results_data, aggregation_function=aggregation_function)

    error_data.to_csv(prepare_directory(output_path) / f'{error_metric_class.name}_{aggregation}.csv')

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(error_data)
    ax.set_title(f'{error_metric_class.full_name} {aggregation}')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{error_metric_class.name} [{error_metric_class.unit}]')
    ax.grid()

    plt.savefig(prepare_directory(output_path) / f'{error_metric_class.name}_{aggregation}.png')
    if show_plot:
        plt.show()

    plt.close()


if __name__ == '__main__':
    results_error_analytics()
