# import common library
import numpy as np
import tensorflow as tf
from tensorflow import keras

import time
import sys

# import custom module
from SimilarNet import SimilarNet
from SimilarNetParametric import SimilarNetParametric

# set args
IS_CONCAT = False
IS_PARAMETRIC = False
if len(sys.argv) >= 2:
    if sys.argv[1].lower() == "concat":
        IS_CONCAT = True
    elif sys.argv[1].lower() == "similarnetparametric":
        IS_PARAMETRIC = True

NORMALIZE_DATA = True
if len(sys.argv) >= 3:
    if sys.argv[2].lower() == "false":
        NORMALIZE_DATA = False

# load raw dataset
datasetStr = "MNIST-CIFAR"

X_train_pair_left = np.load("./data/MNIST-CIFAR/X_train_pair_left.npy")
X_train_pair_right = np.load("./data/MNIST-CIFAR/X_train_pair_right.npy")
y_train_pair = np.load("./data/MNIST-CIFAR/y_train_pair.npy")

X_valid_pair_left = np.load("./data/MNIST-CIFAR/X_valid_pair_left.npy")
X_valid_pair_right = np.load("./data/MNIST-CIFAR/X_valid_pair_right.npy")
y_valid_pair = np.load("./data/MNIST-CIFAR/y_valid_pair.npy")

X_test_pair_left = np.load("./data/MNIST-CIFAR/X_test_pair_left.npy")
X_test_pair_right = np.load("./data/MNIST-CIFAR/X_test_pair_right.npy")
y_test_pair = np.load("./data/MNIST-CIFAR/y_test_pair.npy")

# make embedding model
embedding_model_left = keras.models.Sequential()
embedding_model_left.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid"))
embedding_model_left.add(tf.keras.layers.BatchNormalization())
embedding_model_left.add(tf.keras.layers.ReLU())
embedding_model_left.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

embedding_model_left.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid"))
embedding_model_left.add(tf.keras.layers.BatchNormalization())
embedding_model_left.add(tf.keras.layers.ReLU())
embedding_model_left.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

embedding_model_left.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
embedding_model_left.add(tf.keras.layers.BatchNormalization())
embedding_model_left.add(tf.keras.layers.ReLU())

embedding_model_left.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
embedding_model_left.add(tf.keras.layers.BatchNormalization())
embedding_model_left.add(tf.keras.layers.ReLU())

if IS_CONCAT & NORMALIZE_DATA: embedding_model_left.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x,axis=[-1, -2, -3])))

embedding_model_right = keras.models.Sequential()
embedding_model_right.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid"))
embedding_model_right.add(tf.keras.layers.BatchNormalization())
embedding_model_right.add(tf.keras.layers.ReLU())
embedding_model_right.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

embedding_model_right.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid"))
embedding_model_right.add(tf.keras.layers.BatchNormalization())
embedding_model_right.add(tf.keras.layers.ReLU())
embedding_model_right.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

embedding_model_right.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
embedding_model_right.add(tf.keras.layers.BatchNormalization())
embedding_model_right.add(tf.keras.layers.ReLU())

embedding_model_right.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
embedding_model_right.add(tf.keras.layers.BatchNormalization())
embedding_model_right.add(tf.keras.layers.ReLU())

if IS_CONCAT & NORMALIZE_DATA: embedding_model_right.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x,axis=[-1, -2, -3])))


# comparator model definition
class ComparatorModel(tf.keras.Model):
    def __init__(self, embeddingModel_left, embeddingModel_right, **kwargs):
        super().__init__(**kwargs)

        self.input_ = tf.keras.layers.InputLayer()
        self.embedding_left = embeddingModel_left
        self.embedding_right = embeddingModel_right
        self.flatten_left = tf.keras.layers.Flatten()
        self.flatten_right = tf.keras.layers.Flatten()

        self.similarnet = SimilarNet(activation=SimilarNet.cosine, hetero=True, normalize=NORMALIZE_DATA)
        if IS_PARAMETRIC: self.similarnet = SimilarNetParametric(hetero=True, normalize=NORMALIZE_DATA)
        if IS_CONCAT: self.similarnet = tf.keras.layers.Concatenate()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(8, activation='relu')
        self.output_ = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_A, input_B = inputs[0], inputs[1]
        tensor_A = self.input_(input_A)
        tensor_A = self.embedding_left(tensor_A)
        tensor_A = self.flatten_left(tensor_A)

        tensor_B = self.input_(input_B)
        tensor_B = self.embedding_right(tensor_B)
        tensor_B = self.flatten_right(tensor_B)

        tensor = self.similarnet((tensor_A, tensor_B))

        tensor = self.flatten(tensor)
        tensor = self.dense(tensor)
        tensor = self.output_(tensor)
        return tensor

# train a model
save_seq_code = time.strftime("%y%m%d-%H%M%S")

tf.keras.backend.clear_session()
whole_model = ComparatorModel(embeddingModel_left=embedding_model_left, embeddingModel_right=embedding_model_right)
whole_model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

tensorboard_cb = tf.keras.callbacks.TensorBoard("./logs/run_" + str(save_seq_code))
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
history = whole_model.fit((X_train_pair_left, X_train_pair_right), y_train_pair,
                          validation_data=((X_valid_pair_left, X_valid_pair_right), y_valid_pair),
                          batch_size=32,
                          steps_per_epoch=256,
                          epochs=10000, callbacks = [tensorboard_cb, early_stopping_cb])

accuracy_default = whole_model.evaluate((X_test_pair_left, X_test_pair_right), y_test_pair)[1]

# save model
def saveWholeModel(whole_model, tag, appendix=None):
        dir_name = "./models/" + datasetStr + "/" + datasetStr + "_" + str(tag) + "/"
        whole_model.save(dir_name + "whole.tfmodel")
        
        if appendix != None:
            file = open(dir_name + str(appendix) + ".tag", "w")
            file.close()

compTypeStr = "_concat_"
if whole_model.similarnet.__class__.__name__ == "SimilarNet":
    compTypeStr = "_SimilarNet_"
elif whole_model.similarnet.__class__.__name__ == "SimilarNetParametric":
    compTypeStr = "_SimilarNetParametric_"

normTypeStr = "L2_normalized_"
if NORMALIZE_DATA == False:
    normTypeStr = "not_normalized_"

saveWholeModel(whole_model, str(save_seq_code) + compTypeStr + normTypeStr + str(round(accuracy_default, 4)))
tf.keras.backend.clear_session()