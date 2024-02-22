import logging
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from prediction_system.data_utils.path_manager import prepare_directory
from prediction_system.data_utils.results_data import ResultsData

from utils.exception_logger_decorator import exception_logger

from data import DATA_DIR
logging.basicConfig(level=logging.INFO)


def process_single_result(data_list: list[Path], output_path: Path):
    logging.info(f'Calculating predictions statistics for {data_list[0].name}')

    res = ResultsData.from_csv(data_list[0])
    pred = pd.DataFrame(res.prediction)
    for i, file in enumerate(data_list[1:]):
        data = ResultsData.from_csv(file)
        pred[f'pred_{i}'] = data.prediction.to_list()

    progress_bar = tqdm(total=len(pred), desc='Calculating statistics')

    def calculate_stats(row):
        progress_bar.update(1)
        return pd.Series({
            'prediction': row.mean(),
            'prediction_count': row.count(),
            'prediction_std': row.std(),
            'prediction_min': row.min(),
            'prediction_25%': row.quantile(0.25),
            'prediction_50%': row.median(),
            'prediction_75%': row.quantile(0.75),
            'prediction_max': row.max(),
        })

    stats = pred.apply(calculate_stats, axis=1)

    progress_bar.colour = 'green'
    progress_bar.close()

    logging.info(f'Saving predictions statistics to {output_path}')
    res.to_dataframe().drop(columns=['prediction']).join(stats).to_csv(prepare_directory(output_path.parent) / f'{output_path.name}')


@click.command()
@click.option(
    '--data_path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    multiple=True,
    help='Path to the data files to make approximation'
)
@click.option(
    '--output_path',
    default=DATA_DIR / 'evaluation/models_evaluation/models_results',
    type=click.Path(path_type=Path),
    help='Path to the output directory where to save the results of the approximation'
)
@exception_logger
def results_approximation(
        data_path: list[Path],
        output_path: Path,
):
    """
    Generates predictions statistics for the given data files and saves them to the given output directory.

    Args:
        data_path: Path to the data files. Could be a list of directories or a list of files with .csv extension
        output_path: Path to the output directory. Should be csv file or a directory in case of multiple directories as input
    """
    logging.info(f'Collecting predictions statistics for the given data files')
    if all([file.suffix == '.csv' for file in data_path]):
        process_single_result(data_path, output_path)
    else:
        all_files = set([file.name for dir in data_path for file in dir.iterdir() if file.suffix == '.csv'])

        for file in all_files:
            process_single_result([dir / file for dir in data_path if (dir / file).exists()], output_path / file)


if __name__ == '__main__':
    results_approximation()
