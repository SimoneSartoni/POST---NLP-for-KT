from time import time
from sklearn.linear_model import Perceptron
from Knowledge_Tracing.code.evaluation.batch_generation import *
from Knowledge_Tracing.code.evaluation.predictor import predictor


class perceptron(predictor):
    def __init__(self):
        super().__init__(name="perceptron")
        self.bias = 0.0
        self.encoding_model = None
        self.prediction_model = None

    def train(self, encoding_model, train_set, validation_set):
        self.encoding_model = encoding_model
        self.prediction_model = Perceptron(tol=1e-4, random_state=50)
        similarities, labels = generate_similarities(encoding_model, train_set.problems, train_set.labels, train_set.lengths)
        self.prediction_model.fit(similarities, labels)
        print(self.prediction_model.get_params())

    def compute_predictions(self, test_set):
        t = time()
        similarities, self.labels = generate_similarities(self.encoding_model, test_set.problems, test_set.labels, test_set.lengths)
        self.predictions = self.prediction_model.predict(similarities)
        self.time_to_predict = round((time() - t) / 60, 2)
        print(similarities)
        print(self.predictions)
        print(self.labels)
        return self.labels, self.predictions
