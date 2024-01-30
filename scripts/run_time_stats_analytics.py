from pathlib import Path

import click
import pandas as pd
import matplotlib.pyplot as plt
from utils.exception_logger_decorator import exception_logger

from prediction_system.data_utils.path_manager import prepare_directory

from data import DATA_DIR


@click.command()
@click.option(
    '--time_stats_path',
    type=click.Path(),
    default=DATA_DIR / 'time_stats.csv',
    help='Path to the time stats file'
)
@click.option(
    '--output_path',
    type=click.Path(),
    default=DATA_DIR / 'time_stats',
    help='Path to the output directory'
)
@click.option(
    '--show_plot',
    default=True,
    help='Show plot'
)
@exception_logger
def run_time_stats_analytics(
    time_stats_path: Path,
    output_path: Path,
    show_plot: bool
):
    """
    Runs time stats analytics
    """
    data = pd.read_csv(time_stats_path)

    summary = data[data.category != 'other'].groupby('category').sum()

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(summary.index, summary.time)
    ax.set_title('Time stats')
    ax.set_xlabel('Category')
    ax.set_ylabel('Time (s)')
    ax.grid()
    plt.savefig(prepare_directory(output_path) / 'time_stats.png')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    run_time_stats_analytics()