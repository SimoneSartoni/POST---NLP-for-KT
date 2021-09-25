from time import time

import mlflow
import numpy as np

from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from Knowledge_Tracing.code.experiments.working_experiments.clean_and_load_experiments.load_preprocessed_datasets import load_preprocessed_datasets
from Knowledge_Tracing.code.visualization.histogram import comparison_histogram, histogram2


class user_performances_analysis(experiment):
    def __init__(self):
        super().__init__("user_performances_analysis", "data analysis")

    def run(self, current_experiment):
        initial_t = round(time() / 60.0, 2)
        with mlflow.start_run(experiment_id=current_experiment.experiment_id, run_name=current_experiment.name):
            input_datasets = load_preprocessed_datasets().run(current_experiment, assist_12=False)

            for input_dataset in input_datasets:
                users_avg_scores_distribution, problems_avg_scores_dict, problems_avg_scores_distribution = \
                    self.compute_distributions(input_dataset)
                self.draw_distributions(input_dataset, users_avg_scores_distribution, problems_avg_scores_dict,
                                             problems_avg_scores_distribution)

    @staticmethod
    def compute_distributions(dataset):
        users_avg_scores_distribution = []
        problems_scores = {}
        problems_avg_scores_dict = {}
        problems_avg_scores_distribution = []
        for problem in dataset.problem_id_to_index.keys():
            problems_scores[problem] = []
        for user_problems, user_labels in list(zip(dataset.problems, dataset.labels)):
            users_avg_scores_distribution.append(np.mean(user_labels))
            for problem, label in list(zip(user_problems, user_labels)):
                problems_scores[problem].append(label)
        for problem in dataset.problem_id_to_index.keys():
            problem_avg = np.mean(np.array(problems_scores[problem]))
            problems_avg_scores_dict[problem] = problem_avg
            problems_avg_scores_distribution.append(problem_avg)
        return users_avg_scores_distribution, problems_avg_scores_dict, problems_avg_scores_distribution

    @staticmethod
    def draw_distributions(dataset, users_distr, problems_scores_dict, problems_scores_list):
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/results/" + dataset.prefix + dataset.name\
               + "/"
        histogram2(data=users_distr, label_x="users_average_scores", path=path, bins_range=np.linspace(0, 1, 11))
        histogram2(data=problems_scores_list, label_x="problems average scores", path=path, bins_range=np.linspace(0, 1,
                                                                                                                   11))
        mlflow.log_artifact(path)

