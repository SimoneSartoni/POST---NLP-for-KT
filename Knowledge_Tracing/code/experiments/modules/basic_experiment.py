from abc import abstractmethod, ABCMeta


class basic_experiment(metaclass=ABCMeta):
    def __init__(self):
        self.dataset = None
        self.encode_model = None
        self.prediction_model = None

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def compute_predictions(self):
        pass
