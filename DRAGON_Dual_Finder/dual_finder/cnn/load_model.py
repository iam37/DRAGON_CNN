import numpy as np
import tensorflow as tf

import keras
import platform
from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from keras import backend as K

import glob
import tqdm
from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join
from os.path import exists
from optuna.integration import TFKerasPruningCallback

from create_cnn import ModelCreator

class loadModelClass:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
    def f1_score(self, y_true, y_pred):
        precision = tf.keras.metrics.Precision(thresholds=0.5)
        recall = tf.keras.metrics.Recall(thresholds=0.5)
        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)
        precision_result = pr|ecision.result()
        recall_result = recall.result()
        f1 = 2 * ((precision_result * recall_result) / (precision_result + recall_result + tf.keras.backend.epsilon()))
        precision.reset_states()
        recall.reset_states()
        return f1
    def MCC(self, y_true, y_pred):
        true_positives = tf.keras.metrics.TruePositives()
        true_negatives = tf.keras.metrics.TrueNegatives()
        false_positives = tf.keras.metrics.FalsePositives()
        false_negatives = tf.keras.metrics.FalseNegatives()
        true_positives.update_state(y_true, y_pred)
        true_negatives.update_state(y_true, y_pred)
        false_positives.update_state(y_true, y_pred)
        false_negatives.update_state(y_true, y_pred)
        true_positives_result = true_positives.result()
        true_negatives_result = true_negatives.result()
        false_positives_result = false_positives.result()
        false_negatives_result = false_negatives.result()
        mcc = ((true_positives_result*true_negatives_result) -
               (false_positives_result-false_negatives_result))/np.sqrt((true_positives_result + false_positives_result)*(true_positives_result + false_negatives_result)*(true_negatives_result + false_positives_result)*(true_negatives_result + false_negatives_result))
        return mcc

    # Function for loading the weights of a model trained using a different version of Tensorflow.
    def load_model_diff_version(self, initial_learning_rate):
        model = models.Sequential()
        model.add(layers.Input(shape=(60, 60, 1)))
        model.add(layers.Rescaling(1./255))

        model.add(layers.Conv2D(96, (12, 12), padding='same', strides=(2, 2), activation=tf.nn.leaky_relu))
        model.add(layers.Dropout(0.32523228915885216))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(256, (6, 6), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu))
        model.add(layers.Dropout(0.32523228915885216))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(256, (3,3), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu))
        model.add(layers.Dropout(0.32523228915885216))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        for _ in range(1):
            model.add(layers.Conv2D(384, kernel_size= 3, strides=(1, 1), activation=tf.nn.leaky_relu))
            model.add(layers.Dropout(0.32523228915885216))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation=tf.nn.leaky_relu))
        #model.add(layers.Dropout(0.5))
        model.add(layers.Dropout(0.32523228915885216))
        #model.add(layers.BatchNormalization())
        model.add(layers.Dense(512, activation = 'relu'))
        model.add(layers.Dense(2, activation='sigmoid'))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5.518358e-07,
                                                                      decay_steps=10000,decay_rate=0.9)

        optimized = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimized, loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                               tf.keras.metrics.FalsePositives(name='false positives'),
                               tf.keras.metrics.FalseNegatives(name='false negatives'),
                               tf.keras.metrics.TruePositives(name='true positives'),
                               tf.keras.metrics.TrueNegatives(name='true negatives'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               self.f1_score, self.MCC], run_eagerly=True)

        model.load_weights(self.path_to_model)
        return model

    def load_model(self, initial_learning_rate):
        custom_object = {"ExponentialDecay": tf.keras.optimizers.schedules.ExponentialDecay}
        trained_model = models.load_model(self.path_to_model, custom_objects = custom_object, compile=False)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                      decay_steps=10000,decay_rate=0.9)
        opt = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate)
        trained_model.compile(optimizer = opt, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.F1Score(name = 'f1_score')], run_eagerly=False)
        return trained_model
