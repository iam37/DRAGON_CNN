import click
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import h5py

# IMPORTANT: Ensure this matches the name of your dataset.py file
from dataset import FITSDataset


def center_crop_or_pad_torch(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Crops or pads a 2D PyTorch tensor to the specified square size from the center."""
    h, w = tensor.shape

    # Pad if the image is smaller than the requested size
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
        h, w = tensor.shape

    # Crop if the image is larger
    start_y = h // 2 - size // 2
    start_x = w // 2 - size // 2
    return tensor[start_y:start_y + size, start_x:start_x + size]


def process_single_object(task_args):
    """Worker function to process a single object. Returns a numpy array for HDF5 or None if corrupted."""
    df_index, row, data_dir, bands, cutout_size, device_str = task_args
    channels = []

    for band in bands:
        fits_path = data_dir / str(row[band])
        try:
            tensor_2d = FITSDataset.load_fits_as_tensor(fits_path, device=device_str)
            tensor_2d = center_crop_or_pad_torch(tensor_2d, cutout_size)
            channels.append(tensor_2d)
        except Exception:
            # Signal failure by returning None instead of zero-padding
            return df_index, None

    stacked_tensor = torch.stack(channels, dim=0)

    # Return numpy array for saving to HDF5. cpu() ensures it's off the GPU.
    return df_index, stacked_tensor.cpu().numpy()


@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True, help='Base directory containing FITS files.')
@click.option('--csv-path', type=click.Path(exists=True), required=True, help='Path to the metadata CSV.')
@click.option('--out-dir', type=click.Path(), required=True, help='Output directory for the .pt tensor files.')
@click.option('--bands', multiple=True, default=['g_band', 'i_band', 'r_band'], help='List of columns for channels.')
@click.option('--cutout-size', type=int, default=94, help='Final square size of the cutouts.')
@click.option('--workers', type=int, default=4, help='Number of parallel CPU workers for disk I/O.')
@click.option('--use-gpu/--no-gpu', default=False, help='Flag to push tensor operations to GPU.')
def generate_tensors(data_dir, csv_path, out_dir, bands, cutout_size, workers, use_gpu):
    """Preprocesses FITS files into an HDF5 dataset and generates a clean metadata CSV."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    device_str = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    click.echo(f"Processing up to {len(df)} objects into {len(bands)}-channel tensors...")
    click.echo(f"Using {workers} workers. Compute Device: {device_str.upper()}")

    # Package tasks with their original dataframe index
    tasks = [
        (i, row, data_dir, bands, cutout_size, device_str)
        for i, (_, row) in enumerate(df.iterrows())
    ]

    h5_path = out_dir / "tensors.h5"

    # Open HDF5 file in write mode
    with h5py.File(h5_path, 'w') as h5f:
        max_len = len(df)

        # 1. Pre-allocate the FULL shape immediately (no shape=(0,...))
        # 2. Chunk it by a larger number, like 64 or 128, to optimize batch reading
        dset = h5f.create_dataset(
            "images",
            shape=(max_len, len(bands), cutout_size, cutout_size),
            maxshape=(max_len, len(bands), cutout_size, cutout_size),
            dtype='float32',
            chunks=(64, len(bands), cutout_size, cutout_size)  # <-- THE MAGIC FIX
        )

        current_h5_idx = 0
        successful_indices = []

        # executor.map ensures results are returned in order
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(process_single_object, tasks)

            for df_index, numpy_array in tqdm(results, total=len(tasks)):
                if numpy_array is not None:
                    # 3. Assign directly to the pre-allocated space (NO resizing in loop!)
                    dset[current_h5_idx] = numpy_array
                    successful_indices.append(df_index)
                    current_h5_idx += 1

        # 4. Do one final resize at the very end to trim any dropped/corrupted files
        if current_h5_idx < max_len:
            dset.resize(current_h5_idx, axis=0)

    # Filter out the corrupted rows to ensure perfect alignment
    clean_df = df.iloc[successful_indices].copy()
    clean_csv_path = out_dir / "clean_info.csv"
    clean_df.to_csv(clean_csv_path, index=False)

    click.echo(f"Finished! Packed {len(successful_indices)} tensors sequentially into {h5_path}")
    click.echo(f"Saved aligned metadata to {clean_csv_path} (use this for training!)")


if __name__ == '__main__':
    import torch.multiprocessing as mp

    # Force PyTorch/Python to use 'spawn' instead of 'fork'
    mp.set_start_method('spawn', force=True)
    generate_tensors()