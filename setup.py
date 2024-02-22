from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ml_project',
    version='0.1.0',
    author='Karol Pilot',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'model_evaluation=scripts.run_model_evaluation:model_evaluation',
            'generate_learning_curves=scripts.run_learning_curve_generator:generate_learning_curves',
            'time_stats_analytics=scripts.run_time_stats_analytics:time_stats_analytics',
            'results_error_analytics=scripts.run_results_error_analytics:results_error_analytics',
            'results_approximation=scripts.run_results_approximation:results_approximation',
        ]
    }
)
