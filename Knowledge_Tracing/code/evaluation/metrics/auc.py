from Knowledge_Tracing.code.evaluation.metrics.metrics import metric
from sklearn.metrics import roc_auc_score


class auc(metric):
    """Compute balanced accuracy defined as rateo: (TP + TN) / (TP + TN + FP + FN), equal to:  (TP + TN) / number of labels
    """

    def __init__(self, name="auc"):
        self.name = name
        self.auc = 0.0

    def evaluate(self, labels, predictions, scores):
        self.auc = 0.0
        if len(scores) > 0:
            self.auc = roc_auc_score(labels, scores)
        return self.auc
