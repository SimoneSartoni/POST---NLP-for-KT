from time import time


class evaluator:
    def __init__(self, name, metrics):
        self.name = name
        self.time_to_evaluate = None
        self.performances = {}
        self.metrics = metrics

    def evaluate(self, labels, predictions, models, predictors):
        t = time()
        print(labels)
        print(predictions)
        for predictor in predictors:
            self.performances[predictor.name] = {}
            for model in models:
                self.performances[predictor.name][model.name] = {}
                for metric in self.metrics:
                    self.performances[predictor.name][model.name][metric.name] = metric.evaluate(labels=labels[predictor.name], predictions=predictions[predictor.name][model.name])
        self.time_to_evaluate = round((time() - t) / 60, 2)
        return self.performances
