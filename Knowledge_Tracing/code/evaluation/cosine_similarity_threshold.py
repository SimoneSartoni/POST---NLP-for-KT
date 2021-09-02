from time import time

from sklearn.svm import SVC
from skopt import BayesSearchCV
from Knowledge_Tracing.code.evaluation.batch_generation import *
from Knowledge_Tracing.code.evaluation.predictor import predictor
from Knowledge_Tracing.code.evaluation.evaluator import evaluator
from Knowledge_Tracing.code.evaluation.balanced_accuracy import balanced_accuracy
from skopt.space import Real, Categorical, Integer


class threshold:
    def __init__(self, weight, threshold_value):
        self.weight = weight
        self.threshold_value = threshold_value

    def fit(self):
        return None


class cosine_similarity_threshold(predictor):
    def __init__(self):
        super().__init__(name="cosine_similarity_threshold")
        self.bias = 0.0
        self.encoding_model = None

    def train(self, encoding_model, train_set, validation_set):
        self.encoding_model = encoding_model
        similarities, labels = generate_similarities(encoding_model, train_set.problems, train_set.labels,
                                                     train_set.lengths)
        predictions = {}
        for iterator in range(-10, 10):
            bias = iterator/10.0
            predictions[bias] = []
            for similarity in similarities:
                if similarity >= bias:
                    predictions[bias].append(1.0)
                else:
                    predictions[bias].append(0.0)
        metric = [balanced_accuracy()]
        evaluator_train = evaluator("train_bias", metric)
        self.performances = evaluator_train.train_evaluate(predictions, labels)
        best_performance = 0.0
        self.bias = 0
        for el in self.performances.keys():
            if self.performances[el]["balanced_accuracy"] > best_performance:
                best_performance = self.performances[el]["balanced_accuracy"]
                self.bias = el
        print("best bias is:" + str(self.bias) + ", with performances: " + str(best_performance))

    def compute_predictions(self, test_set):
        t = time()
        similarities, self.labels = generate_similarities(self.encoding_model, test_set.problems, test_set.labels,
                                                          test_set.lengths)
        for similarity in similarities:
            if similarity >= self.bias:
                self.predictions.append(1.0)
            else:
                self.predictions.append(0.0)
        self.time_to_predict = round((time() - t) / 60, 2)
        return self.labels, self.predictions
