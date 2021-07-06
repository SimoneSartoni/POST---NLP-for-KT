from abc import abstractmethod, ABC


class base_model(ABC):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.time_to_build = 0
        self.time_to_train = 0

    @abstractmethod
    def _compute_problem_score(self, problems, corrects, target_problem):
        pass

