import numpy as np
from sklearn.model_selection import RepeatedKFold, GridSearchCV, RandomizedSearchCV

from Knowledge_Tracing.code.evaluation.predictors.cosine_similarity_threshold import cosine_similarity_threshold
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from Knowledge_Tracing.code.evaluation.metrics.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.evaluation.metrics.auc import auc
from Knowledge_Tracing.code.experiments.modules.word2vec import word2vec
from Knowledge_Tracing.code.experiments.modules.baseline_constant import *
from Knowledge_Tracing.code.experiments.working_experiments.clean_and_load_experiments.load_preprocessed_datasets import load_preprocessed_datasets
from Knowledge_Tracing.code.evaluation.evaluator import *
from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from mlflow import *


class optimize_word2vec(experiment):
    def __init__(self):
        super().__init__("optimize_word2vec", "optimization")

    def run(self, current_experiment):
        with start_run(experiment_id=current_experiment.experiment_id, run_name=current_experiment.name):
            input_datasets = load_preprocessed_datasets().run(current_experiment, assist_12=False)
            metrics = [balanced_accuracy(name="balanced_accuracy"), auc(name="auc")]
            experiments = []
            i = 0
            min_count_list = [1, 2, 3, 4, 5, 10, 20, 30]
            window_list = [10, 25, 50, 100, 200]
            vector_size_list = [50, 100, 200, 300, 500]
            sg_list = [0, 1]
            epochs_list = [10, 20, 30, 40]

            for input_dataset in input_datasets:
                experiments.append([])
                for trials in range(0, 1):
                    # x1 = word2vec(dataset=input_dataset, prediction_model=cosine_similarity_threshold(), load=False)
                    x2 = word2vec(dataset=input_dataset, prediction_model=logistic_regressor(), load=False)
                    min_count = min_count_list[np.random.randint(0, len(min_count_list))]
                    window = window_list[np.random.randint(0, len(window_list))]
                    vector_size = vector_size_list[np.random.randint(0, len(vector_size_list))]
                    sg = sg_list[np.random.randint(0, len(sg_list))]
                    epochs = epochs_list[np.random.randint(0, len(epochs_list))]
                    # x1.set_params(**{"min_count": min_count, "window": window, "vector_size": vector_size, "sg": sg,
                                    # "epochs": epochs})
                    x2.set_params(**{"min_count": min_count, "window": window, "vector_size": vector_size, "sg": sg,
                                    "epochs": epochs})

                    # experiments[i].append(x1)
                    experiments[i].append(x2)
                experiments[i].append(baseline_constant(dataset=input_dataset))
                i += 1
            for dataset_index in range(0, len(input_datasets)):
                for exp in experiments[dataset_index]:
                    exp.encode()
                    exp.prediction_train()
                    exp.compute_predictions()
                evaluator_ = evaluator(input_datasets[dataset_index].name, metrics=metrics)
                evaluator_.evaluate(experiments[dataset_index])

        self.time_to_process = round(time() / 60.0, 2)
        print('Time to process vocab: {} mins'.format(self.time_to_process - self.time_to_import))
