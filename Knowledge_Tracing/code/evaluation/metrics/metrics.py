from abc import ABC, abstractmethod


class metric(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def evaluate(self, labels, predictions, scores):
        pass