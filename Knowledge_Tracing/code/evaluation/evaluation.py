import time


class predictor:
    def __init__(self, name, metrics):
        self.name = name
        self.time_to_evaluate = None
        self.performances = dict({})
        self.metrics = metrics

    def evaluate(self, labels, models, predictions):
        t = time()
        for model, prediction in list(zip(*(models, predictions))):
            self.performances[model.name] = dict({})
            for metric in self.metrics:
                self.performances[model.name][metric.name] = metric.evaluate(labels, predictions)
        self.time_to_evaluate = round((time() - t) / 60, 2)
        return self.performances
