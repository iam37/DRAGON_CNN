from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import pathlib
from pathlib import Path
from tqdm import tqdm
import numpy as np

import kornia.augmentation as K
import operator

import shutil
from ipywidgets import Output
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.backend_bases import MouseButton
import glob
import shutil

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]
def load_fits_images(data_path, target_shape=(94, 94)):
    training_images = []
    image_files = []

    #for image_path in tqdm(data_path.glob("**/*.fits")):
    for image_path in tqdm(glob.glob(data_path + "*.fits")):
        with fits.open(image_path, memmap=False) as hdul:
            img = hdul[1].data
            if img is not None and img.shape >= target_shape:
                img = cropND(img, target_shape)
                training_images.append(img)
                image_files.append(str(image_path))

    return training_images, image_files
def display_images_with_buttons(training_images, image_files):
    num_images = min(len(training_images), 49)
    random_offset_selection = np.random.choice(len(training_images), size=num_images, replace=False)

    buttons = []
    grid = []

    def on_button_click(b):
        idx = b.image_index
        img_path = image_files.pop(random_offset_selection[idx])  # Remove from image_files
        training_images.pop(random_offset_selection[idx])  # Remove from training_images

        # Revision to move the images to the "rubbish" catagory for further training.
        """
        if os.path.exists(img_path):
            os.remove(img_path)
            b.style.button_color = 'red'
            b.description = f'{img_path}'
            print(f"Deleted: {img_path}")"""
        move_filepath = "data_preprocessing/training_datasets/rubbish_images/train_data/"
        if not os.path.exists(move_filepath):
            os.makedirs(move_filepath)
        if os.path.exists(img_path):
            shutil.move(img_path, move_filepath)
            b.style.button_color = 'turquoise'
            b.description = f'moved to {move_filepath}'
            print(f"Moved: {img_path} to {move_filepath}")

    for i in range(num_images):
        idx = random_offset_selection[i]
        img = training_images[idx]
        img_name = image_files[idx]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99), cmap='viridis')
        ax.set_title(img_name[-50:], fontsize = 8)
        ax.axis('off')
        plt.close(fig)
        
        button = widgets.Button(description=f'Image {i+1}')
        button.image_index = i
        button.on_click(on_button_click)

        output = widgets.Output()
        with output:
            display(fig)

        grid.append(widgets.VBox([button, output]))

    grid_layout = widgets.GridspecLayout(7, 7, height='auto')
    for i, item in enumerate(grid):
        grid_layout[i // 7, i % 7] = item

    display(grid_layout)
