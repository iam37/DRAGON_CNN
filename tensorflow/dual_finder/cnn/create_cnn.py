import numpy as np
import tensorflow as tf
import keras
from keras import layers, models

class MulticlassPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='multiclass_precision', **kwargs):
        super(MulticlassPrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        for i in range(self.num_classes):
            true_pos = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), tf.float32))
            false_pos = tf.reduce_sum(tf.cast((y_true != i) & (y_pred == i), tf.float32))
            
            self.true_positives.assign_add(tf.tensor_scatter_nd_add(self.true_positives, [[i]], [true_pos]))
            self.false_positives.assign_add(tf.tensor_scatter_nd_add(self.false_positives, [[i]], [false_pos]))
    
    def result(self):
        precisions = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        macro_precision = tf.reduce_mean(precisions)
        return macro_precision
    
    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


class MulticlassRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='multiclass_recall', **kwargs):
        super(MulticlassRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        for i in range(self.num_classes):
            true_pos = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), tf.float32))
            false_neg = tf.reduce_sum(tf.cast((y_true == i) & (y_pred != i), tf.float32))
            
            self.true_positives.assign_add(tf.tensor_scatter_nd_add(self.true_positives, [[i]], [true_pos]))
            self.false_negatives.assign_add(tf.tensor_scatter_nd_add(self.false_negatives, [[i]], [false_neg]))
    
    def result(self):
        recalls = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        macro_recall = tf.reduce_mean(recalls)
        return macro_recall
    
    def reset_stats(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))



class ModelCreator:
    def __init__(self, image_shape, learning_rate, num_classes, importance_score, display_architecture = True):
        self.image_shape = image_shape
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.display_architecture = display_architecture
        self.importance_score = importance_score
    def create_model(self, dropout, display_architecture = True):
        model = models.Sequential()
        model.add(layers.Input(shape=(60, 60, 1)))
        model.add(layers.Rescaling(1./255))
    
        model.add(layers.Conv2D(96, (12, 12), padding='same', strides=(2, 2), activation=tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())
    
        model.add(layers.Conv2D(256, (6, 6), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())
    
        model.add(layers.Conv2D(256, (3,3), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu,  kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())
    
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        for _ in range(1):
            model.add(layers.Conv2D(384, kernel_size= 3, strides=(1, 1), activation=tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
            model.add(layers.Dropout(dropout))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation=tf.nn.leaky_relu,  kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        #model.add(layers.Dropout())
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(512, activation = tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dense(2, activation='softmax'))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,
                                                                      decay_steps=10000,decay_rate=0.9)
    
        optimized = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
        model.compile(optimizer=optimized, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[WeightedPrecision(num_classes=self.num_classes, importance_scores=self.importance_scores),
                               WeightedRecall(num_classes=self.num_classes, importance_scores=self.importance_scores)], run_eagerly=False)
    
        if display_architecture:
            model.summary()
    
        return model
    def create_expanded_model(self, dropout, display_architecture = True):
        model = models.Sequential()
        model.add(layers.Input(shape=self.image_shape))
        model.add(layers.Rescaling(1./255))
    
        model.add(layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())
        model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))
    
        model.add(layers.Conv2D(96, (3, 3), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())
        model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))
    
        model.add(layers.Conv2D(128, (3,3), padding='same', strides=(1, 1), activation=tf.nn.leaky_relu,  kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(256, (3,3), padding='same', strides = (1, 1), activation = tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())


        model.add(layers.Conv2D(384, (3,3), padding = 'same', strides = (1,1), activation = tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(384, (3,3), padding = 'same', strides = (1,1), activation = tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(512, (3,3), padding = 'same', strides = (1,1), activation = tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())
        model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation = tf.nn.leaky_relu, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(0.5))
        print(self.num_classes)
        model.add(layers.Dense(self.num_classes, activation = 'softmax'))
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,
                                                                      decay_steps=10000,decay_rate=0.9)
    
        optimized = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
        model.compile(optimizer=optimized, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.F1Score(name='f1_score')], run_eagerly=False)
    
        if display_architecture:
            model.summary()
    
        return model
        #assert image_shape.shape != (image_shape.shape[0], image_shape.shape[0], 1), "Image shape must have three channels"
