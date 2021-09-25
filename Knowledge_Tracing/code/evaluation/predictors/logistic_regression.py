from time import time

import numpy as np
from mlflow import set_tag
from sklearn.linear_model import LogisticRegression

from Knowledge_Tracing.code.evaluation.batch_generation import *
from Knowledge_Tracing.code.evaluation.predictors.predictor import predictor
from Knowledge_Tracing.code.evaluation.evaluator import evaluator
from Knowledge_Tracing.code.evaluation.metrics.balanced_accuracy import balanced_accuracy


class logistic_regressor(predictor):
    def __init__(self):
        super().__init__(name="logistic_regressor")
        self.regressor = None
        self.encoding_model = None
        self.random_state = 42

    def train(self, encoding_model, train_set):
        self.encoding_model = encoding_model
        x, y = generate_features_encoding(encoding_model, train_set.problems, train_set.labels, train_set.lengths)
        print(x)
        self.regressor = LogisticRegression(penalty="l1", solver="liblinear", random_state=self.random_state).fit(x, y)
        self.predictions = self.regressor.predict(x)

    def compute_predictions(self, test_set):
        t = time()
        similarities, self.labels = generate_features_encoding(self.encoding_model, test_set.problems, test_set.labels,
                                                          test_set.lengths)
        self.predictions = self.regressor.predict(similarities)
        print(self.regressor.predict_proba(similarities))
        for el in self.regressor.predict_proba(similarities):
            self.scores.append(el[1])
        self.time_to_predict = round((time() - t) / 60, 2)
        return self.labels, self.predictions

    def get_serializable_params(self):
        params = self.regressor.get_params()
        return {"name": self.name, "params": params}
