from time import time


class evaluator:
    def __init__(self, name, metrics):
        self.name = name
        self.time_to_evaluate = None
        self.performances = {}
        self.metrics = metrics

    def evaluate(self, labels, models, predictions):
        t = time()
        print(labels)
        print(predictions)
        for model, prediction in list(zip(*(models, predictions))):
            self.performances[model.name] = {}
            for metric in self.metrics:
                self.performances[model.name][metric.name] = metric.evaluate(labels=labels, predictions=predictions[model.name])
        self.time_to_evaluate = round((time() - t) / 60, 2)
        return self.performances
