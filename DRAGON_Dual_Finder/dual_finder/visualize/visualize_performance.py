import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import exists
import sys
import glob


def load_training_history(npy_filepath):
    try:
        history = np.load(npy_filepath, allow_pickle = True)
        history_dict = history.item()
        return history_dict
    except:
        directory, extension = os.path.splitext(npy_filepath)
        assert extension == 'npy', "File must have the .npy extension" 
def plot_training_progress(loss, acc, epochs, training_run = "example", save_filepath = "saved_figures/",
                           recall = None, precision = None, f1_score = None,
                          val_loss = None, val_acc = None, val_recall = None, val_precision = None, 
                           val_f1_score = None):
    if (type(val_loss) == type(None)) or (type(val_acc) == type(None)) or (type(val_recall) == type(None)) or (type(val_precision) == type(None)) or (type(val_f1_score) == type(None)):
        # For reasons yet unknown, the f1 score values are being saved as a nested array of two values. I will continue to investigate this
        # but for now, we will access the second value in both the f1_score and val_f1_score arrays
        f1_score = [entry[1] for entry in f1_score]
        fig, ax = plt.subplots(2, 3, figsize = (20,20))
        ax[0][0].plot(epochs, loss, color = 'midnightblue', label = 'Loss')
        ax[0][0].set_title(f'Loss vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[0][0].set_xlabel("Epochs [Number of iterations]")
        ax[0][0].set_ylabel("Loss Function Value [unitless]")
        ax[0][0].legend()
        
        ax[0][1].plot(epochs, acc, color = 'lightsalmon', label = "Accuracy")
        ax[0][1].set_title(f'Accuracy vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[0][1].set_xlabel("Epochs [Number of iterations]")
        ax[0][1].set_ylabel("Accuracy value [unitless]")
        ax[0][1].legend()

        ax[0][2].plot(epochs, recall, color = 'firebrick', label = "Recall")
        ax[0][2].set_title(f'Recall vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[0][2].set_xlabel("Epochs [Number of iterations]")
        ax[0][2].set_ylabel("Recall value [unitless]")
        ax[0][2].legend()

    
        ax[1][0].plot(epochs, precision, color = 'midnightblue', label = 'Precision', fontsize = 14)
        ax[1][0].set_title(f'Precision vs. Epoch for Training Run: {training_run}')
        ax[1][0].set_xlabel("Epochs [Number of iterations]")
        ax[1][0].set_ylabel("Precision Value [unitless]")
        ax[1][0].legend()
        
        ax[1][1].plot(epochs, f1_score, color = 'lightsalmon', label = "F1 Score")
        ax[1][1].set_title(f'F1 Score vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[1][1].set_xlabel("Epochs [Number of iterations]")
        ax[1][1].set_ylabel("F1 Score [unitless]")
        ax[1][1].legend()
        ax[1][2].axis('off')

        plt.tight_layout()
        plt.savefig(save_filepath + "training_plots.png")
        plt.show()
        plt.close()
        return fig, ax
    else:
        # For reasons yet unknown, the f1 score values are being saved as a nested array of two values. I will continue to investigate this
        # but for now, we will access the second value in both the f1_score and val_f1_score arrays
        f1_score = [entry[1] for entry in f1_score]
        val_f1_score = [entry[1] for entry in val_f1_score]
        fig, ax = plt.subplots(2, 3, figsize = (20,20))
        ax[0][0].plot(epochs, loss, color = 'midnightblue', label = 'Loss', linewidth = 3, alpha = 0.85)
        ax[0][0].plot(epochs, val_loss, color = "lightsalmon", label = 'Validation Loss', linewidth = 3, alpha = 0.85)
        ax[0][0].set_title(f'Loss vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[0][0].set_xlabel("Epochs [Number of iterations]")
        ax[0][0].set_ylabel("Loss Function Value [unitless]")
        ax[0][0].legend()
        
        ax[0][1].plot(epochs, acc, color = 'midnightblue', label = "Accuracy", linewidth = 3, alpha = 0.85)
        ax[0][1].plot(epochs, val_acc, color = 'mediumturquoise', label = "Validation Accuracy", linewidth = 3, alpha = 0.85)
        ax[0][1].set_title(f'Accuracy vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[0][1].set_xlabel("Epochs [Number of iterations]")
        ax[0][1].set_ylabel("Accuracy value [unitless]")
        ax[0][1].legend()

        ax[0][2].plot(epochs, recall, color = 'midnightblue', label = "Recall", linewidth = 3, alpha = 0.85)
        ax[0][2].plot(epochs, val_recall, color = 'firebrick', label = "Validation Recall", linewidth = 3, alpha = 0.85)
        ax[0][2].set_title(f'Recall vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[0][2].set_xlabel("Epochs [Number of iterations]")
        ax[0][2].set_ylabel("Recall value [unitless]")
        ax[0][2].legend()

    
        ax[1][0].plot(epochs, precision, color = 'midnightblue', label = 'Precision', linewidth = 3, alpha = 0.85)
        ax[1][0].plot(epochs, val_precision, color = 'lightsalmon', label = 'Validation Precision', linewidth = 3, alpha = 0.85)
        ax[1][0].set_title(f'Precision vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[1][0].set_xlabel("Epochs [Number of iterations]")
        ax[1][0].set_ylabel("Precision Value [unitless]")
        ax[1][0].legend()
        
        ax[1][1].plot(epochs, f1_score, color = 'midnightblue', label = "F1 Score", linewidth = 3, alpha = 0.85)
        ax[1][1].plot(epochs, val_f1_score, color = 'mediumturquoise', label = "Validation F1 Score", linewidth = 3, alpha = 0.85)
        ax[1][1].set_title(f'F1 Score vs. Epoch for Training Run: {training_run}', fontsize = 14)
        ax[1][1].set_xlabel("Epochs [Number of iterations]")
        ax[1][1].set_ylabel("F1 Score [unitless]")
        ax[1][1].legend()
        
        ax[1][2].axis('off')


        if not exists(save_filepath):
            os.makedirs(save_filepath)
        plt.tight_layout()
        plt.savefig(save_filepath + "training_plots.png")
        plt.show()
        return fig, ax
def plot_grouped_training_progress(training_data, epochs, save_filepath):
    if not exists(save_filepath):
        os.makedirs(save_filepath)
        
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))
    metric_names = ["Accuracy", "Loss", "Recall", "Precision", "F1 Score"]
    colors = ['midnightblue', 'lightsalmon', 'mediumturquoise', 'firebrick']
    line_types = ['-', '--']

    # Unpack data
    for idx, metric in enumerate(metric_names):
        ax = axs[idx // 3, idx % 3]
        for run_idx, run_data in enumerate(training_data):
            for val_idx, (label, values) in enumerate(run_data[metric].items()):
                #print(np.shape(epochs[run_idx]))
                ax.plot(epochs[run_idx], values, linestyle = line_types[val_idx], color = colors[run_idx], label=f"{run_data['label']} {label}")
        ax.set_title(f'{metric} vs. Epochs')
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.legend()

    axs[1, 2].axis('off')  # Turn off the last subplot (if it's unused)
    plt.tight_layout()
    plt.savefig(save_filepath + "combined_training_plots.png")
    plt.show()
class VisualizeOptimization:
    def __init__(self, trial_filepath):
        self.trial_filepath = trial_filepath
    def extract_best_hyperparams(self, param_filepath):
        df = pd.read_csv(param_filepath + "/best_hyperparams_synth_{optimize_score}.csv")
        best_value = df.loc[df['Parameter'] == 'best_value', 'Value'].values
        init_learning_rate = df.loc[df['Parameter'] == 'learning_rate', 'Value'].values
        init_batch_size = df.loc[df['Parameter'] == 'batch_size', 'Value'].values
        num_frozen_layers = df.loc[df['Parameter'] == 'num_frozen_layers', 'Value'].values
        dropout = df.loc[df['Parameter'] == 'dropout', 'Value'].values
        unfreeze_learning_rate = df.loc[df['Parameter'] == 'unfreeze_learning_rate', 'Value'].values
        unfreeze_batch_size = df.loc[df['Parameter'] == 'unfreeze_batch_size', 'Value'].values
        return init_learning_rate, init_batch_size, num_frozen_layers, dropout, unfreeze_learning_rate, unfreeze_batch_size, best_value
    def plot_best_params(self, num_trials = 10):
        learning_rates = []
        init_batch_sizes = []
        num_frozen_layer_array = []
        dropout_rates = []
        unfreeze_learning_rates =[]
        unfreeze_batch_sizes = []
        best_values = []
        for ii in range(num_trials):
            param_filepath = self.trial_filepath + "saved_models_trial_"+str(ii)
            #filename = glob.glob(param_filepath)
            #parts = filename.split(".")
        
            init_learning_rate, init_batch_size, num_frozen_layers, dropout, unfreeze_learning_rate, unfreeze_batch_size, best_value = self.extract_best_hyperparams(param_filepath)
            learning_rates.append(init_learning_rate)
            num_frozen_layer_array.append(num_frozen_layers)
            init_batch_sizes.append(init_batch_size)
            dropout_rates.append(dropout)
            unfreeze_learning_rates.append(unfreeze_learning_rate)
            unfreeze_batch_sizes.append(unfreeze_batch_size)
            best_values.append(best_value)
        
       # We now plot the "best_value" parameter of our optimization run over trial number. We also overlay plots of the 
       # different hyperparameter values per each trial run to visualize how the change in these parameters by the TPE algorithm
       # correlates to the change in the best value.
        trial_number = np.arange(0, num_trials)
        fig, axes = plt.subplots(1, 3, figsize =(10,4))
        axes[0].plot(trial_number, best_values, linewidth = 2, color = 'black', label = "Best Value [1-accuracy]", linestyle = "solid")
        axes[0].set_title("Optimization Score vs. Optimization Trial", fontsize = 8)
        axes[0].set_xlabel("Trial Number", fontsize = 9)
        axes[0].set_ylabel("Optimization Score [unitless]", fontsize = 10)
        axes[0].legend(loc = 0, fontsize = 7)
        
        axes[1].plot(trial_number, learning_rates, color = 'midnightblue', linestyle = "dotted", linewidth = 1, label = "Learning rates")
        axes[1].plot(trial_number, dropout_rates, color = "mediumturquoise", linewidth = 3, linestyle = (0, (10, 3)), label = "Dropout rate")
        axes[1].plot(trial_number, unfreeze_learning_rates, color = "crimson", linewidth = 3, linestyle = (0, (3, 10, 1, 10)), label = "Learning Rate used during Unfreezing")
        axes[1].set_title("Learning and Dropout Rates vs. Optimization Trial", fontsize = 8)
        axes[1].set_xlabel("Trial number", fontsize = 9)
        axes[1].set_ylabel("Hyperparameter value [unitless]", fontsize = 9)
        axes[1].set_yscale('log')
        axes[1].legend(loc = 0, fontsize = 7)
        
        axes[2].plot(trial_number, init_batch_sizes, color = "lightsalmon", linewidth = 3, linestyle = "dashed", label = "Initial (synthetic) batch sizes")
        axes[2].plot(trial_number, num_frozen_layer_array, color = "darkolivegreen", linewidth = 3, linestyle = "dashdot", label = "Number of Frozen Layers")
        axes[2].plot(trial_number, unfreeze_batch_sizes, color = "slategray", linewidth = 3, linestyle = (0, (5, 10)), label = "Batch Size during Unfreezing")
        
        axes[2].set_title("Batch Size and Number of Frozen Layers vs. Optimization Trial", fontsize = 8)
        axes[2].set_xlabel("Trial Number", fontsize = 9)
        axes[2].set_ylabel("Hyperparameter [varying units]", fontsize = 9)
        axes[2].set_yscale('log')
        axes[2].legend(loc = 0, fontsize = 7)
        plt.tight_layout()
        plt.savefig(self.trial_filepath + "plotted_optimization_metrics.png", dpi = 300)
        plt.show()



