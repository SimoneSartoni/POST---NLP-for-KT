import tensorflow as tf


class CumSumLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CumSumLayer, self).__init__(**kwargs)

    def call(self, input_feature):
        output = tf.reduce_sum(input_feature, axis=-1, keepdims=True)
        return output

    def compute_mask(self, input_feature, mask=None):
        if mask is None:
            return None
        return mask


class NegativeLabelLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NegativeLabelLayer, self).__init__(**kwargs)

    def call(self, input_feature):
        output = tf.reduce_sum(input_feature, axis=-1, keepdims=True)
        return output

    def compute_mask(self, input_feature, mask=None):
        if mask is None:
            return None
        return mask