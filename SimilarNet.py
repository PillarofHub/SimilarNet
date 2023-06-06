# SimilarNet class (keras)
# author : Yi Chungheon ( pillarofcode@gmail.com )

# import
import tensorflow as tf


# layer definition
class SimilarNet(tf.keras.layers.Layer):
    # activation functions
    def cosine(z1, z2):
        return z1 * z2

    def bird(z1, z2):
        positive = tf.math.maximum(
            tf.math.maximum(
                tf.math.abs(z1), tf.math.abs(z2)
            ) - 2 * tf.math.abs(z1 - z2)
            , 0)
        negative = tf.math.maximum(
            tf.math.maximum(
                tf.math.abs(z1), tf.math.abs(z2)
            ) - 2 * tf.math.abs(z1 + z2)
            , 0)
        return positive - negative

    # model definition
    def __init__(self, activation=cosine, hetero=False, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.hetero = hetero
        self.normalize = normalize

    def build(self, batch_input_shape):
        b1, b2 = batch_input_shape
        self.norm_axis_X1 = tf.range(1, len(b1.as_list()), 1)
        self.norm_axis_X2 = tf.range(1, len(b2.as_list()), 1)

        super().build(batch_input_shape)

    def call(self, X):
        X1, X2 = X
        if self.normalize:
            X1 = tf.math.l2_normalize(X1, axis=self.norm_axis_X1)
            X2 = tf.math.l2_normalize(X2, axis=self.norm_axis_X2)

        if self.hetero:
            X1 = tf.expand_dims(X1, axis=-1)
            X2 = tf.expand_dims(X2, axis=-2)

        return self.activation(X1, X2)

    def compute_output_shape(self, batch_input_shape):
        b1, b2 = batch_input_shape
        if not self.hetero:
            return b1
        else:
            return tf.TensorShape([b1[0], b1[1], b2[1]])

    def get_config(self):
        config = {
            "activation": self.activation,
            "hetero": self.hetero,
            "normalize": self.normalize
        }
        return config