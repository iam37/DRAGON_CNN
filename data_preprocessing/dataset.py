from astropy.io import fits
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from utils import (
    load_tensor,
    load_data_dir,
    arsinh_normalize
)

import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
mp.set_sharing_strategy("file_system")


class FITSDataset(Dataset):
    """Dataset from FITS files. Pre-caches FITS files as PyTorch tensors to
    improve data_preprocessing load speed."""

    def __init__(
        self,
        data_dir='/dev/null',
        label_col="classes",
        slug=None,  # a slug is a human readable ID
        split=None,  # splits are defined in make_split.py file.
        cutout_size=94,
        normalize=False,  # Whether labels will be normalized.
        transforms=None,  # Supports a list of transforms or a single transform func.
        channels=1,
        load_labels=True
    ):
        # Set data_preprocessing directories
        self.data_dir = Path(data_dir)  # As long as you keep the label csv in one spot with all nested directories
        self.cutouts_path = self.data_dir / "cutouts"
        self.tensors_path = self.data_dir / "tensors"
        self.tensors_path.mkdir(parents=True, exist_ok=True)

        # Initialize image metadata
        self.channels = channels

        # Initializing cutout shape, assuming the shape is roughly square-like.
        self.cutout_shape = (channels, cutout_size, cutout_size)

        # Set requested transforms
        self.normalize = normalize
        self.transform = transforms

        # Define paths
        self.data_info = load_data_dir(self.data_dir, slug, split)
        self.filenames = np.asarray(self.data_info["file_name"])

        # Loading labels if for training, not if for inference.
        if load_labels:
            label_info_path = self.data_dir / "labels.csv"
            if label_info_path.is_file():
                # If a label csv dictionary is provided with strings
                label_df = pd.read_csv(label_info_path)
                self.label_dict = {row["key"]: row["value"] for _, row in label_df.iterrows()}
                self.labels = np.asarray([self.label_dict[v] for v in self.data_info[label_col]])
            else:
                self.labels = np.asarray(self.data_info[label_col])

            self.num_classes = np.unique(self.labels)
        else:
            # generate fake labels of appropriate shape
            self.labels = np.ones((len(self.data_info), len(label_col)))  # Double check
            self.num_classes = 1

        # If we haven't already generated PyTorch tensor files, generate them
        logging.info("Generating PyTorch tensors from FITS files...")
        for filename in tqdm(self.filenames):
            filepath = self.tensors_path / (filename + ".pt")
            if not filepath.is_file():  # If the tensors were not pre-generated, this returns True.
                # All files saved to one cutouts folder.
                # load_path = self.cutouts_path / filename (If we want to maintain cutouts method)
                load_path = self.data_dir / filename
                t = FITSDataset.load_fits_as_tensor(load_path)

                if '/' in filename:  # Flattening out the directory and altering file path.
                    filename = filename.replace('/', '_')
                    filepath = self.tensors_path / (filename + ".pt")

                torch.save(t, filepath)

        # If instead the files are loaded, preload the tensors!
        n = len(self.filenames)
        filepaths = [fl.replace('/', '_') if '/' in fl else fl for fl in self.filenames]  # Flatten
        logging.info(f"Preloading {n} tensors...")
        load_fn = partial(load_tensor, tensors_path=self.tensors_path)
        with mp.Pool(mp.cpu_count()) as p:
            # Load to NumPy, then convert to PyTorch (hack to solve system
            # issue with multiprocessing + PyTorch tensors)
            self.observations = list(
                tqdm(p.imap(load_fn, filepaths), total=n)
            )

        print("Initialization of FITS Dataset Completed.")

    def __getitem__(self, index):
        """Magic method to index into the dataset."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]  # Slice indexing.
        elif isinstance(index, int):
            # If the index is an integer, we proceed as normal and load up our tensor as a data point.
            # We support wrap around functionality
            pt = self.observations[index % len(self.observations)]

            # Get image label.
            label = torch.tensor(self.labels[index % len(self.labels)])

            # Transform the tensor if a transformation is specified.
            if self.transform is not None:
                if hasattr(self.transform, "__len__"):  # If inputted in a list of transforms
                    for transform in self.transform(pt):
                        pt = transform(pt)
                else:  # If inputted a single transform.
                    pt = self.transform(pt)

            # Normalization of images
            if self.normalize:
                pt = arsinh_normalize(pt)

            return pt.squeeze(1), label

    def __len__(self):
        """Return the effective length of the dataset."""
        return len(self.labels)

    @staticmethod
    def load_fits_as_tensor(filename):
        """Open a FITS file and convert it to a Torch tensor."""
        fits_np = fits.getdata(filename, memmap=False)
        return torch.from_numpy(fits_np.astype(np.float32))
