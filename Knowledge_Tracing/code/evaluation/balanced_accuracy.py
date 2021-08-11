from Knowledge_Tracing.code.evaluation.metrics import metric


class balanced_accuracy(metric):
    """Compute balanced accuracy defined as rateo: (TP + TN) / (TP + TN + FP + FN), equal to:  (TP + TN) / number of labels
    """

    def __init__(self, name):
        self.name = name
        self.acc = 0.0

    def evaluate(self, labels, predictions):
        self.acc = 0.0
        for label, prediction in list(zip(labels, predictions)):
            if label == prediction:
                self.acc += 1.0
        self.acc = self.acc / float(len(labels))
        return self.acc
