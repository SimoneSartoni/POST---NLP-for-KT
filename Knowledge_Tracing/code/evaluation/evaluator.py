import json
from time import time
from mlflow import *

class evaluator:
    def __init__(self, name, metrics):
        self.name = name
        self.time_to_evaluate = None
        self.performances = {}
        self.metrics = metrics

    def train_evaluate(self, predictions, labels):
        t = time()
        for el in predictions.keys():
            self.performances[el] = {}
            for metric in self.metrics:
                self.performances[el][metric.name] = metric.evaluate(labels=labels, predictions=predictions[el])
        self.time_to_evaluate = round((time() - t) / 60, 2)
        file = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/performances_train.json"
        # self.save_performances(file)
        return self.performances

    def evaluate(self, experiments):
        t = time()
        for experiment in experiments:
            self.performances[experiment.name] = {}
            for metric in self.metrics:
                self.performances[experiment.name][metric.name] = metric.evaluate(labels=experiment.labels, predictions=experiment.predictions)
        self.time_to_evaluate = round((time() - t) / 60, 2)
        #  self.save_performances()
        return self.performances

    def save_performances(self, file="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/performances"
                                     ".json"):
        set_tags({"performances": self.performances})

