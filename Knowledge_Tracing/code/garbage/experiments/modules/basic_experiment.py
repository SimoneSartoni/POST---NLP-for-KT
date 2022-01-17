import codecs
import os
from abc import ABCMeta
import json
from mlflow import *


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


class basic_experiment(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name
        self.dataset = None
        self.encode_model = None
        self.prediction_model = None
        self.predictions = []
        self.labels = []

    def log_params(self):
        path = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/temporary/" + self.dataset.name
        make_dir(path)
        path = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/temporary/" + self.dataset.name + "/" + self.name
        make_dir(path)
        if self.encode_model:
            params = self.encode_model.get_serializable_params()
            file = path + "/encode_params.json"
            with open(file, "w") as f:
                json.dump(params, f)
        if self.prediction_model:
            params = self.prediction_model.get_serializable_params()
            file = path + "/prediction_params.json"
            with open(file, "w") as f:
                json.dump(params, f)

    def log_predictions(self):
        path = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/temporary/" + self.dataset.name + "/" + self.name
        file = path + "/predictions.json"
        json.dump(self.predictions.tolist(), codecs.open(file, 'w', encoding='utf-8'), separators=(',', ':'),
                  sort_keys=True, indent=4)
        log_artifact(path)
