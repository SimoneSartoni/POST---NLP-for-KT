import numpy as np
from sklearn.model_selection import RepeatedKFold, GridSearchCV, RandomizedSearchCV

from Knowledge_Tracing.code.evaluation.predictors.cosine_similarity_threshold import cosine_similarity_threshold
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from Knowledge_Tracing.code.evaluation.metrics.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.experiments.modules.count_vect import count_vect
from Knowledge_Tracing.code.experiments.modules.baseline_constant import *
from Knowledge_Tracing.code.experiments.working_experiments.clean_and_load_experiments.load_preprocessed_datasets import load_preprocessed_datasets
from Knowledge_Tracing.code.evaluation.evaluator import *
from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from mlflow import *


class optimize_count_vectorizer(experiment):
    def __init__(self):
        super().__init__("optimize_count_vectorizer", "optimization")

    def run(self, current_experiment):
        with start_run(experiment_id=current_experiment.experiment_id, run_name=current_experiment.name):
            input_datasets = load_preprocessed_datasets().run(current_experiment, assist_12=False)
            metrics = [balanced_accuracy(name="balanced_accuracy")]
            experiments = []
            i = 0

            min_df = [1, 2, 3, 5, 10, 20]
            max_df = [25, 50, 100, 200, 1.0]
            binary = [True, False]
            topK = [100]
            shrink = [0]
            normalize = [True]
            similarity = ["cosine"]

            for input_dataset in input_datasets:
                experiments.append([])
                for trials in range(0, 10):
                    x = count_vect(dataset=input_dataset, prediction_model=logistic_regressor(), load=False)
                    min_df_el = min_df[np.random.randint(0, len(min_df))]
                    max_df_el = max_df[np.random.randint(0, len(max_df))]
                    binary_el = binary[np.random.randint(0, len(binary))]
                    topK_el = topK[np.random.randint(0, len(topK))]
                    shrink_el = shrink[np.random.randint(0, len(shrink))]
                    normalize_el = normalize[np.random.randint(0, len(normalize))]
                    similarity_el = similarity[np.random.randint(0, len(similarity))]
                    x.set_params(**{"min_df": min_df_el, "max_df": max_df_el, "binary": binary_el, "topK": topK_el, "shrink": shrink_el,
                                  "normalize": normalize_el, "similarity": similarity_el})
                    experiments[i].append(x)
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
