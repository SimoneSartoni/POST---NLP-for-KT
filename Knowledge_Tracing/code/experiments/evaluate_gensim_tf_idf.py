from random import random

from Knowledge_Tracing.code.evaluation.perceptron import perceptron
from Knowledge_Tracing.code.evaluation.cosine_similarity_threshold import cosine_similarity_threshold
from Knowledge_Tracing.code.evaluation.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.experiments.modules.tf_idf import *
from Knowledge_Tracing.code.experiments.modules.word2vec import *
from Knowledge_Tracing.code.experiments.load_preprocessed_datasets import load_preprocessed_datasets
from Knowledge_Tracing.code.evaluation.evaluator import *
from Knowledge_Tracing.code.experiments.experiment import experiment
from mlflow import *
import mlflow


class evaluate_gensim_tf_idf(experiment):
    def __init__(self):
        super().__init__("evaluate_gensim_and_tf_idf", "encoding evaluation")

    def run(self, current_experiment):
        input_datasets = load_preprocessed_datasets().run()
        metrics = [balanced_accuracy(name="balanced_accuracy")]
        experiments = []
        for input_dataset in input_datasets:
            experiments.append(tf_idf(input_dataset, prediction_model=cosine_similarity_threshold(), load=False))
            experiments.append(word2vec(input_dataset, prediction_model=cosine_similarity_threshold(), load=False))
        for exp in experiments:
            with start_run(experiment_id=current_experiment.experiment_id):
                exp.encode()
                exp.prediction_train()
                exp.compute_predictions()
                exp.log_params()
                exp.log_predictions()
        evaluator_ = evaluator("balanced_accuracy", metrics=metrics)
        performances = evaluator_.evaluate(experiments)

        self.time_to_process = round(time() / 60.0, 2)
        print('Time to process vocab: {} mins'.format(self.time_to_process - self.time_to_import))