# Dealing with Anomalies ML

A machine learning project focused on detecting and handling anomalies.

## Description

TODO: Write a project description

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
    - [Model evaluation](#model-evaluation)
    - [Learning curves](#learning-curves)
    - [Results analysis](#results-analysis)
    - [Times analysis](#times-analysis)
- [Debugging](#debugging)


## Installation

1. Clone the repository:

```bash
git clone https://github.com/lorak0123/dealing-with-anomalies-ml.git
cd dealing-with-anomalies-ml
```

2. Create a virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Usage

### Model evaluation

Script for generating learning curves for a given model or a list of models.

```bash
model_evaluation [OPTIONS]
```

Options:
- `--data_path` (PATH): Path to the data file
- `--output_path` (PATH): Path to the output directory
- `--step` (INTEGER): Steps to use for learning curves
-  `--model` (TEXT): Model to use for prediction. Must be defined in prediction_system/config.py
- `--n_jobs` (INTEGER): Number of parallel jobs to run
- `--help` Show help message and exit.

#### Example

Command:
```bash
model_evaluation --data_path data/data.csv --output_path output --step 10 --step 20 --step 30 --model DECISION_TREE_REGRESSOR --n_jobs 4
```
Output:
```bash
INFO:root:Generating learning curves for DecisionTreeRegressor
100%|███████████████████████████████████████████████| 1084/1084 [00:06<00:00, 172.70it/s] 
100%|███████████████████████████████████████████████| 1074/1074 [00:06<00:00, 166.17it/s]
100%|███████████████████████████████████████████████| 1064/1064 [00:06<00:00, 156.06it/s]
```

### Learning curves

Script for generating learning curves for the results of model evaluation.

```bash
learning_curves [OPTIONS]
```

Options:
- `--input_path` (PATH): Path to the data directory
- `--output_path` (PATH): Path to the data directory
- `--show_plot` (BOOLEAN): Show plot
- `--interpolate` (BOOLEAN): Interpolate data
- `--error_metric` (TEXT): Error metric to use for evaluation. Must be defined in prediction_system/data_utils/error_metrics/__init__.py
- `--help`: Show help message and exit.

#### Example

Command:
```bash
generate_learning_curves --input_path output --output_path output/learning_curves --show_plot True --interpolate False --error_metric MAAPE
```
Output:
```bash
Reading files: 100%|███████████████████████████████████████████████| 3/3 [00:00<00:00, 13.17it/s]
```

### Results analysis

Script for analyzing the results of model.

```bash
results_error_analytics [OPTIONS]
```

Options:
- `--input_file` (PATH): Path to the results data file
- `--output_path` (PATH): Path to the data directory
- `--aggregation` (TEXT): Aggregation function
- `--show_plot` (BOOLEAN): Show plot
- `--error_metric` (TEXT): Error metric to use for evaluation. Must be defined in prediction_system/data_utils/error_metrics/__init__.py
- `--help`: Show help message and exit.

#### Example

Command:
```bash
results_error_analytics --input_file output/DecisionTreeRegressor_10.csv --output_path output/results_analytics --aggregation WEEKLY --show_plot True --error_metric RMSE
```


### Times analysis

Script for analyzing the times taken by different categories of jobs.

```bash
time_stats_analytics [OPTIONS]
```

Options:
- `--time_stats_path` (PATH): Path to the time stats file
- `--output_path` (PATH): Path to the output directory
- `--show_plot` (BOOLEAN): Show plot
- `--help`: Show help message and exit.

#### Example

Command:
```bash
time_stats_analytics --time_stats_path data/time_stats.csv --output_path output/time_stats_analytics --show_plot True
```


## Debugging

To get more information about the execution of the scripts, you can set the `DEBUG` environment variable to `True`:

```bash
export DEBUG=True
```