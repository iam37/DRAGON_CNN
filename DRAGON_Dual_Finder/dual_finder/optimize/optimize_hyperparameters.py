import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna 
import tqdm
from tqdm import tqdm
#import random
import os
import sys
import csv
from os import listdir
from os.path import isfile, join
from os.path import exists
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler
#from ..cnn.create_cnn import ModelCreator
#from ..cnn.train_cnn import DualFinder
sys.path.append("dual_finder/dual_finder/cnn/")
#from .cnn.create_cnn import ModelCreator
from create_cnn import ModelCreator
from train_cnn import DualFinder
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#rom .cnn.train_cnn import DualFinder


class OptimizeHyperparameters:
    def __init__(self, trial_filepath, pretrained_model = None, synth_train_dataset = None, synth_train_labels = None, synth_val_dataset = None, synth_val_labels = None, X_train = None, X_val = None, train_dataset = None, train_labels = None, val_dataset = None, val_labels = None, image_shape = (60,60,1), learning_rate = 5.518358e-07, batch_size = 32):
        self.trial_filepath = trial_filepath
        self.synth_train_dataset = synth_train_dataset
        self.synth_train_labels = synth_train_labels
        self.synth_val_dataset = synth_val_dataset
        self.synth_val_labels = synth_val_labels
        
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.val_dataset = val_dataset
        self.val_labels = val_labels
        
        self.image_shape = image_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.X_val = X_val
        self.X_train = X_train
        self.pretrained_model = pretrained_model
    def create_model(self, num_classes, dropout_rate):
        model_creator = ModelCreator(self.image_shape, self.learning_rate, display_architecture = True)
        self.model = model_creator.create_model(dropout_rate)
    def objective_transfer_learn(self, trial):

        """Hyperparameters to be optimized"""
        learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        num_frozen_layers = trial.suggest_int("num_frozen_layers", 1, 22)
        dropout_percentage = trial.suggest_float('dropout', 0.2, 0.8)

        unfreeze_learning_rate = trial.suggest_float('unfreeze_learning_rate', 1e-8, learning_rate, log = True)
        unfreeze_batch_size = trial.suggest_categorical('unfreeze_batch_size', [16, 32, 64, 128])

        # Make a DualFinder instance. 
        if self.pretrained_model is None:
            dual_finder = DualFinder(train_dataset=self.synth_train_dataset, validation_dataset = self.synth_val_dataset, image_shape= self.image_shape, initial_labels = self.synth_train_labels, validation_labels = self.synth_val_labels, epoch = 10, batchSize = self.batch_size, learningRate = self.learning_rate, num_layers = 1, model_type = "modelB", display_architecture = True if trial.number == 0 else False)
    
            
            #train_labels_one_hot, val_labels_one_hot = dual_finder.encode_labels(self.train_labels, self.val_labels)
            model_filepath = self.trial_filepath + f"saved_models_trial_{trial.number}/"
            
            history_synth, history_frozen, history_unfreeze, history_unfrozen, fully_trained_model = dual_finder.transferLearning(num_frozen_layers, model_filepath, 35, self.train_dataset, self.train_labels, self.val_dataset, self.val_labels, batch_size, learning_rate, dropout_percentage, train_synth = True, save_feature_maps = False)
        else:
            
            model_filepath = self.trial_filepath + f"saved_models_trial_{trial.number}/"
            
            dual_finder = DualFinder(train_dataset=self.train_dataset, validation_dataset = self.val_dataset, image_shape= self.image_shape, initial_labels = self.train_labels, validation_labels = self.val_labels, epoch = 10, batchSize = self.batch_size, learningRate = self.learning_rate, num_layers = 1, model_type = "modelB", display_architecture = True if trial.number == 0 else False)
            #train_labels_one_hot, val_labels_one_hot = dual_finder.encode_labels(self.train_labels, self.val_labels)
            history_frozen, history_unfreeze, history_unfrozen, fully_trained_model = dual_finder.transferLearning(num_frozen_layers, model_filepath, 30, self.train_dataset, self.train_labels, self.val_dataset, self.val_labels, batch_size, learning_rate, dropout_percentage, train_synth = False, save_feature_maps = False, model = self.pretrained_model)
        
        results = fully_trained_model.evaluate(self.val_dataset, val_labels_one_hot)
        accuracy = results[1]
        optimize_score = 1-accuracy # So that our study can work to minimize this result.
        hyperparams = {"learning_rate": learning_rate, "batch_size": batch_size, "num_frozen_layers": num_frozen_layers, "unfreeze_learning_rate": unfreeze_learning_rate, "unfreeze_batch_size": unfreeze_batch_size, "best_value": optimize_score}
        with open(model_filepath + "best_hyperparams_synth_{optimize_score}.csv", mode='w', newline='') as file:
            logging.info(f"Saving hyperparams to {model_filepath}")
            writer = csv.writer(file)
            writer.writerow(['Parameter', 'Value'])  # Writing headers
            for param, value in hyperparams.items():
                writer.writerow([param, value])  # Writing each parameter and its value
        return optimize_score
    def test_study(self, num_trials, mode = "synth"):
        try:
            study = optuna.create_study(direction = "minimize", sampler = TPESampler())
            if mode == "synth":
                study.optimize(self.objective, n_trials=num_trials)
                best_params = study.best_params
                print('Best Hyperparameters:', best_params)
            elif mode == "transfer learn":
                study.optimize(self.objective_transfer_learn, n_trials = num_trials)
                best_params = study.best_params
                print(f"Best Hyperparameters: {best_params}")
        except: 
            assert (mode == "synth") or (mode == "transfer learn"), " 'mode' input is not valid (must be either 'synth' or 'transfer learn')"





