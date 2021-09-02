from abc import abstractmethod, ABCMeta
import json
from mlflow import *


class basic_experiment(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name
        self.dataset = None
        self.encode_model = None
        self.prediction_model = None
        self.predictions = []
        self.labels = []

    def save_predictions(self):
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/" + self.dataset.name + "/predictions.json"
        with open(file, "w") as f:
            json.dump(self.predictions, f)

    def log_params(self):
        params = dict(vars(self.encode_model).items())
        log_text(json.dumps(params))
        params = dict(vars(self.prediction_model).items())
        log_text(json.dumps(params))

    def log_predictions(self):
        log_text(json.dumps(self.predictions))
