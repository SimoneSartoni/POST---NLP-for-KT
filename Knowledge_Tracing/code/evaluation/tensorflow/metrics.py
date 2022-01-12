import tensorflow as tf


class BinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self):
        super(BinaryAccuracy, self).__init__(name="BinaryAccuracy")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(BinaryAccuracy, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class AUC(tf.keras.metrics.AUC):
    def __init__(self):
        super(AUC, self).__init__(name="AUC")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(AUC, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class Precision(tf.keras.metrics.Precision):
    def __init__(self):
        super(Precision, self).__init__(name="Precision")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(Precision, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class Recall(tf.keras.metrics.Recall):
    def __init__(self):
        super(Recall, self).__init__(name="Recall")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(Recall, self).update_state(y_true=y_true,
                                         y_pred=y_pred,
                                         sample_weight=sample_weight)


class SensitivityAtSpecificity(tf.keras.metrics.SensitivityAtSpecificity):
    def __init__(self):
        super(SensitivityAtSpecificity, self).__init__(name="SensitivityAtSpecificity")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(SensitivityAtSpecificity, self).update_state(y_true=y_true,
                                                           y_pred=y_pred,
                                                           sample_weight=sample_weight)


class SpecificityAtSensitivity(tf.keras.metrics.SpecificityAtSensitivity):
    def __init__(self):
        super(SpecificityAtSensitivity, self).__init__(name="SpecificityAtSensitivity")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(SpecificityAtSensitivity, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class FalseNegatives(tf.keras.metrics.FalseNegatives):
    def __init__(self):
        super(FalseNegatives, self).__init__(name="FalseNegatives")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(FalseNegatives, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class FalsePositives(tf.keras.metrics.FalsePositives):
    def __init__(self):
        super(FalsePositives, self).__init__(name="FalsePositives")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(FalsePositives, self).update_state(y_true=y_true,
                                                 y_pred=y_pred,
                                                 sample_weight=sample_weight)


class TrueNegatives(tf.keras.metrics.TrueNegatives):
    def __init__(self):
        super(TrueNegatives, self).__init__(name="TrueNegatives")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(TrueNegatives, self).update_state(y_true=y_true,
                                                y_pred=y_pred,
                                                sample_weight=sample_weight)


class TruePositives(tf.keras.metrics.TruePositives):
    def __init__(self):
        super(TruePositives, self).__init__(name="TruePositives")

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(TruePositives, self).update_state(y_true=y_true,
                                                y_pred=y_pred,
                                                sample_weight=sample_weight)


class ColdStartBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, window_size=30):
        self.window_size = window_size
        super(ColdStartBinaryAccuracy, self).__init__(name="ColdStartBinaryAccuracy"+str(window_size))

    def update_state(self, y_true, y_pred, sample_weight=None):
        print("y_true")
        print(tf.shape(y_true).numpy())
        print(tf.shape(y_pred).numpy())
        print(tf.shape(sample_weight).numpy())
        if tf.shape(y_true)[1] > self.window_size:
            print("true")
            y_true_2, y_pred_2 = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
            if sample_weight:
                sample_weight_2 = sample_weight[:, 0:self.window_size]
            else:
                sample_weight_2 = sample_weight
        super(ColdStartBinaryAccuracy, self).update_state(y_true=y_true_2, y_pred=y_pred_2, sample_weight=sample_weight_2)


class ColdStartAUC(tf.keras.metrics.AUC):
    def __init__(self, window_size=30):
        self.window_size = window_size
        super(ColdStartAUC, self).__init__(name="ColdStartAUC" + str(window_size))

    def update_state(self, y_true, y_pred, sample_weight=None):
        if tf.shape(y_true)[1] > self.window_size:
            print("true")
            y_true_2, y_pred_2 = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
            if sample_weight:
                sample_weight_2 = sample_weight[:, 0:self.window_size]
            else:
                sample_weight_2 = sample_weight
        super(ColdStartAUC, self).update_state(y_true=y_true_2, y_pred=y_pred_2, sample_weight=sample_weight_2)


class ColdProblemsBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, cold_items):
        self.cold_items = cold_items
        super(ColdProblemsBinaryAccuracy, self).__init__(name="ColdProblemsBinaryAccuracy")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
        sample_weight = sample_weight[:, 0:self.window_size]
        super(ColdProblemsBinaryAccuracy, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class ColdProblemsAUC(tf.keras.metrics.AUC):
    def __init__(self, cold_items):
        self.cold_items = cold_items
        super(ColdProblemsAUC, self).__init__(name="ColdProblemsAUC")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
        sample_weight = sample_weight[:, 0:self.window_size]
        super(ColdProblemsAUC, self).update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
