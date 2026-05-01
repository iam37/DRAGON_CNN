from astropy.io import fits
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import h5py

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils import (
    load_tensor,
    load_data_dir,
    discover_devices
)

import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
mp.set_sharing_strategy("file_system")

def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class FITSDataset(Dataset):
    """Dataset from FITS files. Can load legacy .pt files or stream from a fast HDF5 container."""

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
            force_reload=False,
            n_workers=1,
            expand_factor=1
    ):
        # Set data directories
        self.data_dir = Path(data_dir)
        self.cutouts_path = self.data_dir / "cutouts"
        self.tensors_path = self.data_dir / "tensors"
        self.tensors_path.mkdir(parents=True, exist_ok=True)

        # Discover devices for tensor generation
        device = discover_devices()

        # Initialize image metadata
        self.channels = channels
        self.cutout_shape = (channels, cutout_size, cutout_size)
        self.normalize = normalize
        self.transform = transforms
        self.expand_factor = expand_factor

        # Define paths and load dataframe
        # IMPORTANT: If using HDF5, make sure load_data_dir points to clean_info.csv
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
            self.labels = np.ones(len(self.data_info), dtype=int)
            self.num_classes = 1

        # HDF5 initialization variables
        self.use_h5 = False
        self.h5_path = None
        self.h5_file = None
        self.h5_images = None

        # --- LEGACY VS NEW FORMAT ROUTING ---
        if "file_name" in self.data_info.columns:
            logging.info("Legacy 'file_name' column detected. Routing to legacy logic with individual .pt files.")
            self.filenames = np.asarray(self.data_info["file_name"])
            self.tensor_filepaths = []

            if force_reload:
                logging.info("Force reload on, regenerating legacy tensors...")

            for filename in tqdm(self.filenames, desc="Checking/Generating Legacy Tensors"):
                flattened_filename = filename.replace('/', '_')
                filepath = self.tensors_path / f"{flattened_filename}.pt"
                self.tensor_filepaths.append(str(filepath))

                # On-the-fly generation if missing or forced
                if not filepath.is_file() or force_reload:
                    if self.cutouts_path.is_dir():
                        load_path = self.cutouts_path / filename
                    else:
                        load_path = self.data_dir / filename

                    t = FITSDataset.load_fits_as_tensor(load_path, device)
                    torch.save(t, filepath)

        elif "object_id" in self.data_info.columns:
            logging.info("New 'object_id' column detected. Activating fast HDF5 logic.")
            self.use_h5 = True
            self.h5_path = self.tensors_path / "tensors.h5"

            if not self.h5_path.is_file():
                raise FileNotFoundError(
                    f"HDF5 dataset not found at {self.h5_path}. Please run generate_tensors.py first.")

            # --- NEW: Map CSV rows to exact HDF5 indices ---
            if 'h5_index' not in self.data_info.columns:
                logging.warning("No 'h5_index' column found! Assuming 1:1 mapping. "
                                "If this is a shuffled/balanced split, your labels WILL be scrambled!")
                self.h5_indices = np.arange(len(self.data_info))
            else:
                self.h5_indices = np.asarray(self.data_info['h5_index'])
        else:
            raise KeyError("Metadata CSV must contain either a 'file_name' (legacy) or 'object_id' (new) column.")

        logging.info("Initialization of FITS Dataset Completed.")

        self.sampler = None
        if dist.is_available() and dist.is_initialized():
            self.sampler = DistributedSampler(self, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    def _lazy_init_h5(self):
        """Initializes the HDF5 file lazily when a PyTorch background worker asks for it."""
        if self.h5_file is None:
            # swmr=True enables Single-Writer Multiple-Reader, which is ideal for PyTorch DataLoaders
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=1024 ** 2 * 512)
            self.h5_images = self.h5_file['images']

            max_requested_idx = np.max(self.h5_indices)
            actual_max_idx = self.h5_images.shape[0] - 1

            if max_requested_idx > actual_max_idx:
                # Log the warning so you are aware of the mismatch
                logging.warning(
                    f"Dataset mismatch: CSV requests up to index {max_requested_idx}, "
                    f"but HDF5 only goes up to {actual_max_idx}. Clamping indices to prevent crashes."
                )
                # Clip any out-of-bounds indices to the maximum valid index
                self.h5_indices = np.clip(self.h5_indices, 0, actual_max_idx)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):

            idx = index % len(self.labels)

            if self.use_h5:
                self._lazy_init_h5()
                true_h5_idx = int(self.h5_indices[idx])
                pt_np = self.h5_images[true_h5_idx]
                pt = torch.from_numpy(pt_np)
            else:
                filepath = self.tensor_filepaths[idx]
                pt_np = load_tensor(filepath, tensors_path=self.tensors_path, as_numpy=True)
                pt = torch.from_numpy(pt_np)

            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return pt.squeeze(1), label
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    def __len__(self):
        return len(self.labels) * self.expand_factor

    def __del__(self):
        """Gracefully close the HDF5 file handle upon garbage collection."""
        if self.use_h5 and self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass

    def get_sampler(self):
        return self.sampler

    @staticmethod
    def load_fits_as_tensor(filename, device="cpu"):
        """Open a FITS file and convert it to a Torch tensor, hunting for the correct HDU."""
        fits_np = None

        try:
            fits_np = fits.getdata(filename, memmap=False)
        except OSError as e:
            logging.error(f"ERROR: {filename} is empty or corrupted. Shutting down")
            raise e
        except Exception:
            pass

        # Fallback: Hunt for the 2D data array if getdata fails
        if fits_np is None or not hasattr(fits_np, 'shape') or len(fits_np.shape) < 2:
            try:
                with fits.open(filename, memmap=False) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and hasattr(hdu.data, 'shape') and len(hdu.data.shape) >= 2:
                            fits_np = hdu.data
                            break
                    else:
                        logging.error(f"ERROR: No valid 2D image array found in {filename}.")
                        raise ValueError(f"No valid image array in {filename}")
            except OSError as e:
                logging.error(f"ERROR: {filename} is empty or corrupted. Shutting down")
                raise e

        # Replace NaNs and convert to float32 tensor
        fits_np = np.nan_to_num(fits_np, nan=0.0)
        tensor = torch.from_numpy(fits_np.astype(np.float32))

        # Move to requested device
        if device == 'cuda' or (isinstance(device, str) and device.startswith('cuda')):
            tensor = tensor.to(device)

        return tensor