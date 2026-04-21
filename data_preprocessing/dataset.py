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
            label_col="class",
            slug=None,
            split=None,
            cutout_size=94,
            normalize=False,
            transforms=None,
            channels=3,
            load_labels=True,
            num_classes=None,
            n_workers=1,
            expand_factor=1
    ):
        # Set data directories
        self.data_dir = Path(data_dir)
        self.tensors_path = self.data_dir / "tensors"

        if not self.tensors_path.exists():
            raise FileNotFoundError(
                f"Tensors directory not found at {self.tensors_path}. Please generate tensors first.")

        # Initialize image metadata
        self.channels = channels
        self.cutout_shape = (channels, cutout_size, cutout_size)
        self.normalize = normalize
        self.transform = transforms
        self.expand_factor = expand_factor

        # Define paths and load dataframe
        self.data_info = load_data_dir(self.data_dir, slug, split)

        # Loading labels
        if load_labels:
            label_info_path = self.data_dir / "labels.csv"
            if label_info_path.is_file():
                label_df = pd.read_csv(label_info_path)
                self.label_dict = {row["key"]: row["value"] for _, row in label_df.iterrows()}
                self.labels = np.asarray([self.label_dict[v] for v in self.data_info[label_col]])
            else:
                self.labels = np.asarray(self.data_info[label_col])

            self.num_classes = len(np.unique(self.labels)) if num_classes is None else num_classes
        else:
            self.labels = np.ones((len(self.data_info), 1))
            self.num_classes = 1

        # --- LEGACY VS NEW FORMAT ROUTING ---
        if "file_name" in self.data_info.columns:
            logging.info("Legacy 'file_name' column detected. Routing to legacy filepath logic.")
            self.filenames = np.asarray(self.data_info["file_name"])
            # Replicate your original flattening logic
            self.tensor_filepaths = []
            for fl in self.filenames:
                flattened_filename = fl.replace('/', '_')
                # Assuming .pt extension since we removed inline FITS generation
                self.tensor_filepaths.append(str(self.tensors_path / f"{flattened_filename}.pt"))

        elif "object_id" in self.data_info.columns:
            logging.info("New 'object_id' column detected. Routing to multi-band filepath logic.")
            self.object_ids = np.asarray(self.data_info["object_id"])
            self.tensor_filepaths = [str(self.tensors_path / f"{obj_id}.pt") for obj_id in self.object_ids]

        else:
            raise KeyError("Metadata CSV must contain either a 'file_name' (legacy) or 'object_id' (new) column.")

        # Preload the tensors!
        n = len(self.tensor_filepaths)
        logging.info(f"Preloading {n} PyTorch tensors into memory...")

        load_fn = partial(load_tensor, as_numpy=True)

        with mp.Pool(min(n_workers, mp.cpu_count())) as p:
            self.observations = list(
                tqdm(p.imap(load_fn, self.tensor_filepaths), total=n)
            )
        self.observations = [torch.from_numpy(x) for x in self.observations]

        logging.info("Initialization of Dataset Completed.")

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
