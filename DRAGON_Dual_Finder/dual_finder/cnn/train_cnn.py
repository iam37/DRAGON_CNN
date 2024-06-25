import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
import platform
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, accuracy_score
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
from load_model import loadModelClass
from extract_feature_maps import FeatureExtractor
import warnings
import random

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DualFinder:
    def __init__(self, train_dataset, validation_dataset, image_shape, initial_labels, validation_labels, epoch, batchSize, learningRate, num_classes, model_type, importance_score, display_architecture = True, evaluation_set = None, evaluation_labels = None):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.initial_labels = initial_labels
        self.validation_labels = validation_labels
        self.epoch = epoch
        self.learningRate = learningRate
        self.image_shape = image_shape

        self.evaluation_set = evaluation_set
        self.evaluation_labels = evaluation_labels
        self.batchSize = batchSize
        self.num_classes = num_classes
        self.model_type = model_type
        self.display_architecture = display_architecture
        self.importance_score = importance_score

    def make_one_hot_labels(self, y_train, y_val):
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(y_train)
        val_labels_encoded = label_encoder.transform(y_val)

        train_labels_one_hot = to_categorical(train_labels_encoded)
        val_labels_one_hot = to_categorical(val_labels_encoded)
        train_labels_one_hot = train_labels_one_hot.astype('float32')
        val_labels_one_hot = val_labels_one_hot.astype('float32')
        return train_labels_one_hot, val_labels_one_hot
    """
    Converts training and validation labels to one-hot encoded format.
    
    Parameters:
    y_train (ndarray): Array of training labels.
    y_val (ndarray): Array of validation labels.
    
    Returns:
    tuple: Tuple containing one-hot encoded training and validation labels.
    """

    def freezeLayers(self, model, num_layers_to_freeze, num_conv_layers):
        #for layer in model.layers[1:len(model.layers)-2-num_layers_to_freeze]: # subtract the two fully connected layers.
        for layer in model.layers[2:num_layers_to_freeze]:
            if isinstance(layer, layers.Conv2D):
                layer.trainable = False
        #for layer in model.layers[-(len(model.layers) - num_layers_to_freeze):-1]:
            #layer.trainable = False
        return model
        
    def update_current_epoch(self, epoch):
        self.current_epoch = epoch


    def create_feature_callback(self, feature_map_saver, batch_freq = 100):
         # Custom callback to save feature maps every 5 batch sizes while training the model.
        class FeatureMapCallback(Callback):
            def __init__(self, feature_map_saver, batch_freq=100):
                super().__init__()
                self.feature_map_saver = feature_map_saver  # An instance of your class with save_feature_maps method
                self.batch_freq = batch_freq
        
            def on_batch_end(self, batch, logs=None):
                if (batch + 1) % self.batch_freq == 0:  # +1 to make it work on every 5th batch starting from 1
                    self.feature_map_saver.extract_feature_maps(batch_num=batch)
        feature_instance = FeatureMapCallback(feature_map_saver, batch_freq = 100)
        return feature_instance
    def encode_labels(self, initial_labels, validation_labels):
        #label_mapping = {'empty_sky': 0, 'star':1, 'single_AGN': 2, 'offset_AGN': 3,'dual_AGN': 4}
        label_mapping = {'empty_sky': 0, 'single_AGN': 1, 'dual_AGN': 2, 'merger': 3}
        print(initial_labels)
        print(validation_labels)
        if isinstance(initial_labels, np.ndarray):
            print("Converting to list")
            initial_labels = initial_labels.flatten().tolist()
        if isinstance(validation_labels, np.ndarray):
            print("Converting to list")
            validation_labels = validation_labels.flatten().tolist()
            
        train_labels_numeric = [label_mapping[label] for label in initial_labels]
        train_labels_numeric = np.asarray(train_labels_numeric)
        train_labels_one_hot = to_categorical(train_labels_numeric)
        train_labels_one_hot = train_labels_one_hot.astype('float32')
        val_labels_numeric = [label_mapping[label] for label in validation_labels]
        val_labels_numeric = np.asarray(val_labels_numeric)
        val_labels_one_hot = to_categorical(val_labels_numeric)
        val_labels_one_hot = val_labels_one_hot.astype('float32')
        return train_labels_one_hot, val_labels_one_hot
    # label_mapping = {'empty_sky': 0, 'star': 1, 'single_AGN': 2, 'offset_AGN': 3, 'dual_AGN': 4}
    """def encode_labels(self, initial_labels, validation_labels):
        label_mapping = {'star': 0, 'single_AGN': 1}
        print(initial_labels)
        print(validation_labels)
        if isinstance(initial_labels, np.ndarray):
            print("Converting to list")
            initial_labels = initial_labels.flatten().tolist()
        if isinstance(validation_labels, np.ndarray):
            print("Converting to list")
            validation_labels = validation_labels.flatten().tolist()
            
        train_labels_numeric = [label_mapping[label] for label in initial_labels]
        train_labels_numeric = np.asarray(train_labels_numeric)
        
        val_labels_numeric = [label_mapping[label] for label in validation_labels]
        val_labels_numeric = np.asarray(val_labels_numeric)
        
        return train_labels_numeric, val_labels_numeric"""

        
    def trainCNN(self, dropout_rate = 0.32523228915885216, save_feature_maps = True, model_filepath = "../saved_dual_finder_models/"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.class_names = ["empty_sky", "single_AGN", "dual_AGN", "merger"]
        train_labels, val_labels = self.encode_labels(self.initial_labels, self.validation_labels)
        class_indices = np.argmax(train_labels, axis = 1)
        #class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
        class_weightsDict = dict(enumerate(class_weights))
        print(f"train_labels shape: {train_labels.shape}")
        print(f"val_labels shape: {val_labels.shape}")
        print(f"Unique train labels: {np.unique(train_labels)}")
        print(f"Unique val labels: {np.unique(val_labels)}")
    
        model_creator = ModelCreator(self.image_shape, self.learningRate, self.num_classes, self.importance_score, display_architecture=self.display_architecture)
        model = model_creator.create_expanded_model(dropout_rate)
        model.build((None,) + self.image_shape)
        model.summary()
    
        if not exists(model_filepath):
            os.makedirs(model_filepath)
    
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath + "_checkpoint_training", monitor='val_f1_score', save_weights_only=False, verbose=0)
    
        if save_feature_maps:
            logging.info(f"save_feature_maps == {save_feature_maps}, WILL save feature maps")
            total_indices = np.random.permutation(len(self.validation_dataset))
            random_permutation = total_indices[:1]
            print(f"Feature map will feature: {self.validation_labels[random_permutation]}")
    
            random_images = self.validation_dataset[random_permutation]
            print(f"Shape of randomly selected image: {np.shape(random_images)}")
    
            feature_map_output_filepath = "dual_finder/dual_finder/cnn/feature_maps/" + model_filepath
            if not exists(feature_map_output_filepath):
                os.makedirs(feature_map_output_filepath)
            feature_extractor = FeatureExtractor(model, random_images, feature_map_output_filepath)
            feature_map_callback = self.create_feature_callback(feature_extractor)
            callback_array = [checkpoint_callback, feature_map_callback]
        else:
            logging.info(f"'save_feature_maps' == {save_feature_maps}, NOT saving feature maps")
            callback_array = [checkpoint_callback]

        print(f"train_images shape: {self.train_dataset.shape}")
        print(f"train_labels shape: {train_labels.shape}")
        print(f"val_images shape: {self.validation_dataset.shape}")
        print(f"val_labels shape: {val_labels.shape}")
        history = model.fit(self.train_dataset, train_labels, epochs=self.epoch, verbose=1, validation_data=(self.validation_dataset, val_labels), class_weight=class_weightsDict, callbacks=callback_array, shuffle=True, use_multiprocessing=True)
    
        np.save(model_filepath + "/saved_history", history.history)
        model.save(model_filepath + "/saved_model_" + str(self.epoch) + ".h5")
        return history, model

    def predict(self, model, dataset, filepaths, true_labels):
        # Predict the classes and confidence scores
        predictions = model.predict(dataset)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
    
        # Get the images from the dataset
        if isinstance(dataset, tf.data.Dataset):
            images = list(dataset.as_numpy_iterator())
            images = np.array([img[0] for img in images])  # assuming the dataset yields (image, label) tuples
        else:
            images = dataset
        num_images = min(len(images), 25)
        random_indices = np.random.choice(len(images), num_images, replace=False)
        selected_images = images[random_indices]
        selected_filepaths = filepaths[random_indices]
        selected_predictions = predicted_classes[random_indices]
        selected_confidences = confidence_scores[random_indices]
        selected_true_labels = true_labels[random_indices]
    
        # Plot the images with their predicted classes and confidence scores
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(selected_images[i], cmap="viridis", vmin=np.percentile(selected_images[i], 1), vmax=np.percentile(selected_images[i], 99))
            #print(len(selected_filepaths[i]))
            predicted_label = self.class_names[selected_predictions[i]]
            confidence = selected_confidences[i]
            plt.title(f"True: {selected_true_labels[i]}, pred: {predicted_label}\n{confidence:.2f}", fontsize = 6)
            print(f"{selected_filepaths[i][:-20]} with predicted label: {predicted_label} and confidence: {confidence}")
        plt.tight_layout()
        save_filepath = "saved_test_predictions/"
        if not exists(save_filepath):
            os.makedirs(save_filepath)
        plt.savefig(save_filepath+"randomly_selected_images.png")
        plt.show()

        
    def transferLearning(self, num_layers_to_freeze, model_filepath, newEpochs, new_train_data, new_train_labels, new_val_data, new_val_labels, newBatch, newLearningRate, dropout_rate, model = None, save_feature_maps = False, newClassWeightsDict = None):
        print(model_filepath)
        pretrained_model = model
        num_fc_layers_to_freeze = 0
        transfer_model = self.freezeLayers(pretrained_model, num_layers_to_freeze, num_fc_layers_to_freeze)
        logging.info(f"INITIATE TRANSFER LEARNING: {num_layers_to_freeze} frozen")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=newLearningRate,
                                                                      decay_steps=10000,decay_rate=0.9)

        optimized = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
        train_labels_one_hot, val_labels_one_hot = self.encode_labels(new_train_labels, new_val_labels)
        
        class_indices = np.argmax(val_labels_one_hot, axis = 1) #for creating a dictionary of class weights meant to prevent overfitting the dataset
        class_weights = compute_class_weight('balanced', classes = np.unique(class_indices), y = class_indices)
        class_weights_dict = dict(enumerate(class_weights))

        transfer_model.compile(optimizer=optimized, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name = 'accuracy'), tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.F1Score(name = 'f1_score')], run_eagerly = False)

        transfer_model.summary()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath +"_checkpoint_transfer_learning",save_weights_only=False,verbose=0)
        total_indices = np.random.permutation(len(new_val_data))
        if save_feature_maps:
            random_permutation = total_indices[:1]
            try: 
                random_images = new_val_data[random_permutation]
                print(f"Shape of randomly selected image: {np.shape(random_images)}")
            except:
                assert isinstance(new_val_data, np.ndarray), "All datasets must be numpy arrays for this model type!"
                
            feature_map_output_filepath_frozen = "dual_finder/dual_finder/cnn/feature_maps/" + model_filepath+"_frozen"
            if not exists(feature_map_output_filepath_frozen):
                os.makedirs(feature_map_output_filepath_frozen)
            feature_extractor_frozen = FeatureExtractor(transfer_model, random_images, feature_map_output_filepath_frozen)
            feature_map_callback_frozen = self.create_feature_callback(feature_extractor_frozen)
            callback_array = [cp_callback, feature_map_callback_frozen]
        else:
            callback_array = [cp_callback]
        

        history_1 = transfer_model.fit(new_train_data, train_labels_one_hot, batch_size = newBatch, validation_data = (new_val_data, val_labels_one_hot), verbose = 1, epochs = 10, callbacks = callback_array, class_weight = class_weights_dict, shuffle = True, use_multiprocessing = True)
        np.save(model_filepath + "/saved_history_frozen", history_1.history)

        """After 10 epochs of training on completely frozen model, we will gradually unfreeze the frozen layers, train, and then we will train for a couple of epochs on the final unfrozen model"""
        """Implementing gradual unfreezing for transfer learning stage""" #this may help with stability issues.
        epochs_count = 0
        logging.info(f"Gradually unfreezing {num_layers_to_freeze} layers")
        histories_2 = []
        for i in range(num_layers_to_freeze, 0, -1):
            #if isinstance(transfer_model.layers[len(transfer_model.layers)-i], layers.Conv2D):
            if isinstance(transfer_model.layers[i], layers.Conv2D):
               #transfer_model.layers[len(transfer_model.layers)-i].trainable = True
               transfer_model.layers[i].trainable = True
            if save_feature_maps:
                feature_map_output_filepath_unfreeze = "dual_finder/dual_finder/cnn/feature_maps/" + model_filepath+"_unfreeze"
                if not exists(feature_map_output_filepath_unfreeze):
                    os.makedirs(feature_map_output_filepath_unfreeze)
                feature_extractor_unfreeze = FeatureExtractor(transfer_model, random_images, feature_map_output_filepath_unfreeze)
                feature_map_callback_unfreeze = self.create_feature_callback(feature_extractor_unfreeze)
                callback_array_unfreeze = [cp_callback, feature_map_callback_unfreeze]
            else:
                callback_array_unfreeze = [cp_callback]
            history_2 = transfer_model.fit(new_train_data, train_labels_one_hot, batch_size = newBatch, validation_data = (new_val_data, val_labels_one_hot), class_weight = class_weights_dict, verbose = 1, epochs = 1, callbacks = callback_array_unfreeze, shuffle = True, use_multiprocessing = True)
            histories_2.append(history_2.history)
            np.save(model_filepath + f"/saved_history_unfreeze_{i}", history_2.history)
            logging.info(f"Current epoch: {epochs_count}")
            epochs_count+=1
        #np.save(model_filepath+"/saved_history_unfreeze", np.asarray(histories_2))
        difference = newEpochs - epochs_count
        logging.info("Training for additional " + str(difference) + " epochs.")
        feature_map_output_filepath_unfrozen = "dual_finder/dual_finder/cnn/feature_maps/" + model_filepath+"_unfrozen"
        if save_feature_maps:
            if not exists(feature_map_output_filepath_unfrozen):
                os.makedirs(feature_map_output_filepath_unfrozen)
            feature_extractor_unfrozen = FeatureExtractor(transfer_model, random_images, feature_map_output_filepath_unfrozen)
            feature_map_callback_unfrozen = self.create_feature_callback(feature_extractor_unfrozen)
            callback_array_unfrozen = [cp_callback, feature_map_callback_unfrozen]
        else:
            callback_array_unfrozen = [cp_callback]

        history_3 = transfer_model.fit(new_train_data, train_labels_one_hot, batch_size = 64, validation_data = (new_val_data, val_labels_one_hot), epochs = difference, callbacks = callback_array_unfrozen, class_weight = class_weights_dict, shuffle = True, use_multiprocessing = True)
        transfer_model.save(model_filepath + "/retrained_transfer_learn_model" + str(self.epoch) + ".h5")
        np.save(model_filepath + "/saved_history_unfrozen", history_3.history)
        return history_1, history_2, history_3, model

    def scoring(self, model, X, y):
        y_predicted = model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        return accuracy

    def trainFromCheckpoint(self, model_checkpoint_filepath, model_filepath, newEpochs):
        class_names = ['single AGN', 'double AGN']
        image_shape = (94, 94, 1)

        
        #Another method of numerical labeling that ensures that 'double AGN' is the positive class
        label_mapping = {'single AGN': 0, 'double AGN': 1}
        train_labels_numeric = [label_mapping[label] for label in self.initial_labels]
        train_labels_one_hot = to_categorical(train_labels_numeric)
        train_labels_one_hot = train_labels_one_hot.astype('float32')
        val_labels_numeric = [label_mapping[label] for label in self.validation_labels]
        val_labels_one_hot = to_categorical(val_labels_numeric)
        val_labels_one_hot = val_labels_one_hot.astype('float32')
        class_indices = np.argmax(train_labels_one_hot, axis = 1) #for creating a dictionary of class weights meant to prevent overfitting the dataset
        class_weights = compute_class_weight('balanced', classes = np.unique(class_indices), y = class_indices)
        class_weightsDict = dict(enumerate(class_weights))
        #with tf.device("/GPU:0"):
        cp_callback_retrain = tf.keras.callbacks.ModelCheckpoint(filepath=model_filepath,save_weights_only=False,verbose=1)
        historyCallback = modelCheckpoint.fit(self.train_dataset, train_labels_one_hot, batch_size = 64, verbose = 1, validation_data = (self.validation_dataset, val_labels_one_hot), class_weight = class_weightsDict, epochs = newEpochs, callbacks = cp_callback_retrain, shuffle = True, use_multiprocessing = True)
        modelCheckpoint.save(model_filepath+"/_real_model", overwrite = True)
        return self, historyCallback