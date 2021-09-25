from time import time
from abc import abstractmethod, ABCMeta


class predictor(metaclass=ABCMeta):
    def __init__(self, name="base_predictor"):
        self.name = name
        self.time_to_predict = None
        self.scores = []
        self.predictions = []
        self.labels = []

    @abstractmethod
    def train(self, encoding_model, train_set, validation_set):
        pass

    @abstractmethod
    def compute_predictions(self, dataset, models=[]):
        pass

    @abstractmethod
    def get_serializable_params(self):
        pass
