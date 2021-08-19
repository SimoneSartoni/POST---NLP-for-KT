from abc import abstractmethod, ABCMeta


class experiment(metaclass=ABCMeta):
    def __init__(self, name, class_of_method):
        self.name = name
        self.class_of_method = class_of_method
        self.time_to_import = 0
        self.time_to_process = 0
        self.time_to_train = 0
        self.time_to_evaluate = 0
        self.datasets = []
        self.models = []
        self.metrics = []
        self.predictors = []

    @abstractmethod
    def run(self):
        pass