import codecs
import json
from time import time

import pandas as pd
from mlflow import *

import csv


class evaluator:
    def __init__(self, name, metrics):
        self.name = name
        self.time_to_evaluate = None
        self.performances = {}
        self.metrics = metrics
        self.table_csv = None


    def train_evaluate(self, predictions, labels, scores=None):
        t = time()
        for el in predictions.keys():
            self.performances[el] = {}
            for metric in self.metrics:
                self.performances[el][metric.name] = metric.evaluate(labels=labels, predictions=predictions[el],
                                                                     scores=scores)
        self.time_to_evaluate = round((time() - t) / 60, 2)
        # self.save_performances()
        return self.performances

    def evaluate(self, experiments):
        t = time()
        table_csv = []
        for experiment in experiments:
            table_row = [experiment.dataset.name + ":"]
            if experiment.encode_model:
                table_row.append(experiment.encode_model.get_serializable_params())
            else:
                table_row.append("not available or relevant encode model")
            if experiment.prediction_model:
                table_row.append(experiment.prediction_model.name)
                scores = experiment.prediction_model.scores
            else:
                table_row.append("not available or relevant prediction model")
                scores = []
            metrics_dict = {}
            for metric in self.metrics:
                metrics_dict[metric.name] = metric.evaluate(labels=experiment.labels,
                                                            predictions=experiment.predictions, scores=scores)
            table_row.append(metrics_dict)
            table_csv.append(table_row)
        self.table_csv = table_csv

        self.time_to_evaluate = round((time() - t) / 60.0, 2)
        self.log_table_csv(experiment.dataset.name)
        return self.performances

    def save_performances(self):
        path = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/temporary/" + self.name
        file = path + "_performances.json"
        file_csv = path + "_performances.csv"
        with open(file_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.table_csv)
        json.dump(self.performances, codecs.open(file, 'w', encoding='utf-8'), separators=(',', ':'),
                  sort_keys=True,
                  indent=4)
        log_artifact(file)

    def log_table_csv(self, name):
        df = pd.DataFrame(data=self.table_csv, columns=["dataset_name", "encode_params", "prediction_params", "metrics"])
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/temporary/" + name + "_results.csv"
        df.to_csv(path, sep="\n")
        log_artifact(path)
