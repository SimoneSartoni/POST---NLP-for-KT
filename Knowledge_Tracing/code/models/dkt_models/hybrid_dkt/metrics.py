import tensorflow as tf


class BinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(BinaryAccuracy, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(AUC, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(Precision, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(Recall, self).update_state(y_true=y_true,
                                         y_pred=y_pred,
                                         sample_weight=sample_weight)


class SensitivityAtSpecificity(tf.keras.metrics.SensitivityAtSpecificity):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(SensitivityAtSpecificity, self).update_state(y_true=y_true,
                                                           y_pred=y_pred,
                                                           sample_weight=sample_weight)


class SpecificityAtSensitivity(tf.keras.metrics.SpecificityAtSensitivity):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(SpecificityAtSensitivity, self).update_state(y_true=y_true,
                                                           y_pred=y_pred,
                                                           sample_weight=sample_weight)


class FalseNegatives(tf.keras.metrics.FalseNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(FalseNegatives, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class FalsePositives(tf.keras.metrics.FalsePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(FalsePositives, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class TrueNegatives(tf.keras.metrics.TrueNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(TrueNegatives, self).update_state(y_true=y_true,
                                                y_pred=y_pred,
                                                sample_weight=sample_weight)


class TruePositives(tf.keras.metrics.TruePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super(TruePositives, self).update_state(y_true=y_true,
                                                y_pred=y_pred,
                                                sample_weight=sample_weight)


class ColdStartBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, window_size=30):
        self.window_size = window_size
        super(ColdStartBinaryAccuracy, self).__init__()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
        super(BinaryAccuracy, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class ColdStartAUC(tf.keras.metrics.AUC):
    def __init__(self, window_size=30):
        self.window_size = window_size
        super(ColdStartAUC, self).__init__()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
        super(AUC, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
