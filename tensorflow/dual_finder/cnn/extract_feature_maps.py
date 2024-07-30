import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import os
from os.path import exists
import glob
from matplotlib.animation import FuncAnimation
import imageio
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor:
    """
    Extracts feature maps for an inputted randonly-selected image during the training of a DualFinder instance model.

    This class allows for the extraction of feature maps from specific layers during model training. Feature maps
    are saved to a specified directory after being visualized.

    :param model: The model from which to extract features.
    :type model: tf.keras.Model
    :param image_batch: A batch of images to process.
    :type image_batch: np.array
    :param output_filepath: Directory path to save the extracted feature maps images.
    :type output_filepath: str
    """

    def __init__(self, model, image_batch, output_filepath):
        self.model = model
        self.image_batch = image_batch
        self.output_filepath = output_filepath
        #self.batch_num = batch_num

    def create_feature_map_model(self):
        """
        Creates a new model to output the feature maps from the first convolutional layer of the original model.

        This method modifies the architecture of the tf.keras.Model model to output directly after the first convolutional
        layer, allowing visualization of features right after initial image processing steps.

        :returns: A new tf.keras.Model modified to output feature maps from the first convolutional layer.
        :rtype: tf.keras.Model
        """
        self.feature_map_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[1].output)

    def extract_feature_maps(self, batch_num = 0):
        """
        Extracts feature maps by having the `feature_map_model` predict the class of the selected image and selecting the feature map 
        entry to the predicted array. Saves .png images of feature maps to specified save filepath.

        :returns: None
        :rtype: None
        """
       
        self.batch_num = batch_num
        if not exists(self.output_filepath):
            os.makedirs(self.output_filepath)
        feature_extract_model = self.create_feature_map_model()
        
        # Extract the feature map for the batch of images for this particular epoch
        feature_maps = self.feature_map_model.predict(self.image_batch, verbose = 0)
        feature_maps = np.asarray(feature_maps)
        batch_size = np.shape(self.image_batch)[0]

        rows = 4
        cols = 4
       
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
        
        for i in range(rows):
            for j in range(cols):
                if rows == 1:
                    axs[j].imshow(feature_maps[0, :, :, i*cols + j - 1], cmap='hot')
                    axs[j].axis('off')
                else:
                    axs[i, j].imshow(feature_maps[0, :, :, i*cols + j], cmap='hot')
                    axs[i, j].axis('off')
            #plt.colorbar()
        plt.savefig(f'{self.output_filepath}/batch_num_{batch_num}.png', bbox_inches='tight')
        plt.close(fig)

def create_gif(feature_map_filepath, output_gif_name):
    feature_map_images = []
    output_directory = os.path.dirname(output_gif_name)

    # Function required by FuncAnimate 
    def update(frame_index, img_list, img_plot):
        img_plot.set_array(img_list[frame_index])
        return [img_plot]

    
    if not exists(output_directory):
        logging.info(f"Making directory: {output_directory}")
        os.makedirs(output_directory)

    
    if "modelB" in feature_map_filepath:
        feature_map_synth_filepath = feature_map_filepath
        feature_map_frozen_filepath = feature_map_filepath + "_frozen/"
        feature_map_unfreeze_filepath = feature_map_filepath +"_unfreeze/"
        feature_map_unfrozen_filepath = feature_map_filepath +"_unfrozen/"
        
        for feature_map in sorted(glob.glob(feature_map_synth_filepath+"*.png")):
            feature_map_images.append(imageio.imread(feature_map))
        for feature_map in sorted(glob.glob(feature_map_frozen_filepath+"*.png")):
            feature_map_images.append(imageio.imread(feature_map))
        for feature_map in sorted(glob.glob(feature_map_unfreeze_filepath+"*.png")):
            feature_map_images.append(imageio.imread(feature_map))
        for feature_map in sorted(glob.glob(feature_map_unfrozen_filepath+"*.png")):
            feature_map_images.append(imageio.imread(feature_map))
            
        try:
            logging.info(f"Animating {len(feature_map_images)} images")
            fig, ax = plt.subplots()
            img_plot = ax.imshow(feature_map_images[0], animated=True)
            ax.axis('off')
            ani = FuncAnimation(fig, update, frames=len(feature_map_images), fargs=(feature_map_images, img_plot), blit=True)
            ani.save(output_gif_name, writer='pillow', fps=5)
            logging.info(f"GIF saved successfully at {output_gif_name}")
            
        except Exception as FailedGIFException:
            logging.info(f"Failed to save GIF: {FailedGIFException}")
            
    else:
        for feature_map in sorted(glob.glob(feature_map_filepath+"*.png"), key=lambda x: int(x.split('_')[1])):
            feature_map_images.append(imageio.imread(feature_map))
        try:
            fig, ax = plt.subplots()
            img_plot = ax.imshow(feature_map_images[0], animated=True)
            ax.axis('off')
            ani = FuncAnimation(fig, update, frames=len(feature_map_images), fargs=(feature_map_images, img_plot), blit=True)
            ani.save(output_gif_name, writer='pillow', fps=5)
            logging.info(f"GIF saved successfully at {output_gif_name}")
            
        except Exception as FailedGIFException:
            logging.info(f"Failed to save GIF: {FailedGIFException}")
    
