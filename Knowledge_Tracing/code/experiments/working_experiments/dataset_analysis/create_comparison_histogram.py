from time import time

import mlflow

from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from Knowledge_Tracing.code.experiments.working_experiments.clean_and_load_experiments.load_preprocessed_datasets import load_preprocessed_datasets
from Knowledge_Tracing.code.visualization.histogram import comparison_histogram


class create_comparison_histogram(experiment):
    def __init__(self):
        super().__init__("comparison_histogram", "data analysis")

    def run(self, current_experiment):
        initial_t = round(time() / 60.0, 2)
        with mlflow.start_run(experiment_id=current_experiment.experiment_id, run_name=current_experiment.name):
            input_datasets = load_preprocessed_datasets().run(current_experiment)
            data = []
            labels = []
            for input_dataset in input_datasets:
                data.append(input_dataset.lengths)
                labels.append(input_dataset.name[0:5])
            comparison_histogram(data=data, labels=labels, bin_size=5, name="comparison_histogram", path="/Knowledge_Tracing/temporary/")
