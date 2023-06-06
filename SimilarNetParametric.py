# SimilarNet class (keras)
# author : Yi Chungheon ( pillarofcode@gmail.com )

# import
import tensorflow as tf


# layer definition
# layer definition
class SimilarNetParametric(tf.keras.layers.Layer):
    # activation functions
    def pcosine(z1, z2, alpha):
        ret = z1 * z2
        pos = tf.keras.backend.relu(ret)
        neg = -alpha * tf.keras.backend.relu(-ret)
        return pos + neg

    # model definition
    def __init__(self, activation=pcosine, hetero=False, normalize=True, alpha_initializer="ones",
                 alpha_regularizer=None, alpha_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.hetero = hetero
        self.normalize = normalize

        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        self.alpha_regularizer = tf.keras.regularizers.get(alpha_regularizer)
        self.alpha_constraint = tf.keras.constraints.get(alpha_constraint)

    def build(self, input_shape):
        b1, b2 = input_shape
        self.norm_axis_X1 = tf.range(1, len(b1.as_list()), 1)
        self.norm_axis_X2 = tf.range(1, len(b2.as_list()), 1)

        param_shape = list(b1[1:])
        if self.hetero: param_shape = list(tf.TensorShape([b1[0], b1[1], b2[1]])[1:])
        self.alpha = self.add_weight(
            shape=param_shape,
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, X):
        X1, X2 = X
        if self.normalize:
            X1 = tf.math.l2_normalize(X1, axis=self.norm_axis_X1)
            X2 = tf.math.l2_normalize(X2, axis=self.norm_axis_X2)

        if self.hetero:
            X1 = tf.expand_dims(X1, axis=-1)
            X2 = tf.expand_dims(X2, axis=-2)

        return self.activation(X1, X2, self.alpha)

    def compute_output_shape(self, input_shape):
        b1, b2 = input_shape
        if not self.hetero:
            return b1
        else:
            return tf.TensorShape([b1[0], b1[1], b2[1]])

    def get_config(self):
        config = {
            "activation": self.activation,
            "hetero": self.hetero,
            "normalize": self.normalize,
            "alpha_initializer": tf.keras.initializers.serialize(self.alpha_initializer),
            "alpha_regularizer": tf.keras.regularizers.serialize(self.alpha_regularizer),
            "alpha_constraint": tf.keras.constraints.serialize(self.alpha_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))