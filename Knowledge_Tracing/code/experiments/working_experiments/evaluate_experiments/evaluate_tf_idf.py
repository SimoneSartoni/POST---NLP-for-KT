from Knowledge_Tracing.code.evaluation.predictors.cosine_similarity_threshold import cosine_similarity_threshold
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from Knowledge_Tracing.code.evaluation.metrics.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.evaluation.metrics.auc import auc
from Knowledge_Tracing.code.experiments.modules.tf_idf import *
from Knowledge_Tracing.code.experiments.modules.baseline_constant import *
from Knowledge_Tracing.code.experiments.working_experiments.clean_and_load_experiments.load_preprocessed_datasets import load_preprocessed_datasets
from Knowledge_Tracing.code.evaluation.evaluator import *
from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from mlflow import *


class evaluate_tf_idf(experiment):
    def __init__(self):
        super().__init__("evaluate_gensim_and_tf_idf", "encoding evaluation")

    def run(self, current_experiment):
        with start_run(experiment_id=current_experiment.experiment_id, run_name=current_experiment.name):
            input_datasets = load_preprocessed_datasets().run(current_experiment)
            metrics = [balanced_accuracy(name="balanced_accuracy"), auc(name="auc")]
            experiments = []
            i = 0
            for input_dataset in input_datasets:
                experiments.append([])
                experiments[i].append(tf_idf(input_dataset, prediction_model=cosine_similarity_threshold(), load=False))
                experiments[i].append(tf_idf(input_dataset, prediction_model=logistic_regressor(), load=False))
                experiments[i].append(baseline_constant(input_dataset))
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
