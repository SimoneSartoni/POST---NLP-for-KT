import tensorflow as tf


class ColdStartLoss:
    def __init__(self, window_size=30):
        self.window_size = window_size

    def compute_loss(self, y_true, y_pred):
        y_true, y_pred = y_true[:, 0:self.window_size], y_pred[:, 0:self.window_size]
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)

