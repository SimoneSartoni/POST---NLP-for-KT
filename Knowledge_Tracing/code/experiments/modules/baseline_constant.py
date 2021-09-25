import numpy as np

from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment


class baseline_constant(basic_experiment):
    def __init__(self, dataset):
        super().__init__(name="baseline_constant"+dataset.name)
        self.dataset = dataset
        self.best_constant_prediction = 0.0
        self.encode_model = None
        self.prediction_model = None
        self.predictions = []
        self.labels = []

    def encode(self):
        return

    def prediction_train(self):
        labels_train = 0.0
        for correct in self.dataset.train_set.labels:
            labels_train += correct[-1]
        labels_mean = labels_train / len(self.dataset.train_set.labels)
        if labels_mean >= 0.5:
            self.best_constant_prediction = 1.0
        else:
            self.best_constant_prediction = 0.0

    def compute_predictions(self):
        if self.best_constant_prediction == 1.0:
            self.predictions = np.ones(shape=self.dataset.test_set.number_of_users, dtype=np.float)
        else:
            self.predictions = np.zeros(shape=self.dataset.test_set.number_of_users, dtype=np.float)
        self.labels = []
        for correct in self.dataset.test_set.labels:
            self.labels.append(correct[-1])
