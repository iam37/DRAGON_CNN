import click
import pandas as pd
import torch
import numpy as np
from astropy.io import fits
from pathlib import Path
from tqdm import tqdm


def center_crop_or_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Crops or pads a 2D numpy array to the specified square size from the center."""
    h, w = image.shape

    # Pad if the image is smaller than the requested size
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
        h, w = image.shape

    # Crop if the image is larger
    start_y = h // 2 - size // 2
    start_x = w // 2 - size // 2
    return image[start_y:start_y + size, start_x:start_x + size]


@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True, help='Base directory containing FITS files.')
@click.option('--csv-path', type=click.Path(exists=True), required=True, help='Path to the metadata CSV.')
@click.option('--out-dir', type=click.Path(), required=True, help='Output directory for the .pt tensor files.')
@click.option('--bands', multiple=True, default=['g_band', 'i_band', 'r_band'],
              help='List of columns representing the channels/bands.')
@click.option('--cutout-size', type=int, default=94, help='Final square size of the cutouts.')
def generate_tensors(data_dir, csv_path, out_dir, bands, cutout_size):
    """Preprocesses FITS files into stacked PyTorch tensors."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    click.echo(f"Processing {len(df)} objects into {len(bands)}-channel tensors...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        object_id = row['object_id']
        channels = []

        for band in bands:
            # Assuming the CSV contains relative paths to the FITS files
            fits_path = data_dir / str(row[band])

            try:
                with fits.open(fits_path) as hdul:
                    # Adjust index if your data is not in the primary HDU [0]
                    img_data = hdul[0].data.astype(np.float32)

                    # Ensure uniform size
                    img_data = center_crop_or_pad(img_data, cutout_size)
                    channels.append(img_data)
            except Exception as e:
                click.echo(f"\nError loading {fits_path} for object {object_id}: {e}")
                # You might want to handle missing bands gracefully here (e.g., append zeros)
                channels.append(np.zeros((cutout_size, cutout_size), dtype=np.float32))

        # Stack into (C, H, W) and convert to PyTorch tensor
        stacked_array = np.stack(channels, axis=0)
        tensor = torch.from_numpy(stacked_array)

        # Save tensor using the object_id
        save_path = out_dir / f"{object_id}.pt"
        torch.save(tensor, save_path)

    click.echo(f"Finished! Tensors saved to {out_dir}")


if __name__ == '__main__':
    generate_tensors()