from astropy.io import fits
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils import (
    load_tensor,
    load_data_dir,
    arsinh_normalize,
    discover_devices
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
        load_labels=True,
        num_classes=None,
        force_reload=False,
        n_workers=1,
        expand_factor=1
    ):
        # Set data_preprocessing directories
        self.data_dir = Path(data_dir)  # As long as you keep the label csv in one spot with all nested directories
        self.cutouts_path = self.data_dir / "cutouts"
        self.tensors_path = self.data_dir / "tensors"
        self.tensors_path.mkdir(parents=True, exist_ok=True)

        # Initialize image metadata
        self.channels = channels

        device = discover_devices()

        # Initializing cutout shape, assuming the shape is roughly square-like.
        self.cutout_shape = (channels, cutout_size, cutout_size)

        # Set requested transforms
        self.normalize = normalize
        self.transform = transforms
        self.expand_factor = expand_factor

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

            # Declare number of classes automatically
            self.num_classes = np.unique(self.labels) if num_classes is None else num_classes
        else:
            # generate fake labels of appropriate shape
            self.labels = np.ones((len(self.data_info), len(label_col)))  # Double check
            self.num_classes = 1

        # If we haven't already generated PyTorch tensor files, generate them
        logging.info("Generating PyTorch tensors from FITS files...")
        if force_reload:
            logging.info("Force reload on, regenerating...")

        for filename in tqdm(self.filenames):
            flattened_filename = filename.replace('/', '_')  # Flattening out the directory and altering file path.
            filepath = self.tensors_path / (flattened_filename + ".pt")
            if not filepath.is_file() or force_reload:  # If the tensors were not pre-generated, this returns True.
                # All files saved to one cutouts folder.
                if self.cutouts_path.is_dir():
                    load_path = self.cutouts_path / filename  # (If we want to maintain cutouts method)
                else:
                    load_path = self.data_dir / filename

                # Loading and saving tensor to flattened name.
                t = FITSDataset.load_fits_as_tensor(load_path, device)
                torch.save(t, filepath)

        # If instead the files are loaded, preload the tensors!
        n = len(self.filenames)
        logging.info("Preloading PyTorch tensors before transfer...")
        filepaths = [fl.replace('/', '_') if '/' in fl else fl for fl in self.filenames]  # Flatten
        load_fn = partial(load_tensor, tensors_path=self.tensors_path)

        with mp.Pool(min(n_workers, mp.cpu_count())) as p:
            # Load to NumPy, then convert to PyTorch (hack to solve system
            # issue with multiprocessing + PyTorch tensors)
            self.observations = list(
                tqdm(p.imap(load_fn, filepaths), total=n)
            )
        logging.info("Initialization of FITS Dataset Completed.")

        self.sampler = None
        if dist.is_available() and dist.is_initialized():
            self.sampler = DistributedSampler(self, num_replicas=dist.get_world_size(), rank=dist.get_rank())

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
                    for transform in self.transform:
                        pt = transform(pt)
                else:  # If inputted a single transform.
                    pt = self.transform(pt)

            # Normalization of images
            if self.normalize:
                pt = arsinh_normalize(pt)

            return pt.squeeze(1), label
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    def __len__(self):
        """Return the effective length of the dataset."""
        return len(self.labels) * self.expand_factor

    def get_sampler(self):
        """Return the sampler for DistributedSampler"""
        return self.sampler

    @staticmethod
    def load_fits_as_tensor(filename, device="cpu"):
        """Open a FITS file and convert it to a Torch tensor."""
        try:
            fits_np = fits.getdata(filename, memmap=False)
        except OSError as e:
            logging.error(f"ERROR: {filename} is empty or corrupted. Shutting down")
            raise e

        # Replace NaNs with the specified value
        fits_np = np.nan_to_num(fits_np, nan=0)

        tensor = torch.from_numpy(fits_np.astype(np.float32))
        if device == 'cuda':
            tensor = tensor.to(device)

        return tensor
