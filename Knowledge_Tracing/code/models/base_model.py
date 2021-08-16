from abc import abstractmethod, ABCMeta


class base_model(metaclass=ABCMeta):
    def __init__(self, name, class_of_method):
        self.name = name
        self.class_of_method = class_of_method
        self.time_to_build = 0
        self.time_to_train = 0

    @abstractmethod
    def compute_problem_score(self, input_problems, corrects, target_problem, default_value):
        pass

