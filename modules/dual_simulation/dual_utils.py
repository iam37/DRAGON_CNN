import sep
import multiprocessing as mp
import numpy as np

from tqdm import tqdm


def crop_center(img: np.ndarray, cropx: int, cropy: int):
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)

    return img[starty:starty + cropy, startx:startx + cropx, ...]


def write_rotation_test_coordinates(
    quasar_name: str,
    separation_arcsec: float,
    ra_center: float,
    dec_center: float,
    fltrs: list[str],
    n_samples: int = 200,
):
    """
    Generate coordinates on a circle around a central RA/Dec.
    """
    # Convert separation to degrees
    r = separation_arcsec / 3600.0

    # Sample angles in radians
    angles_deg = np.random.uniform(0.0, 360.0, n_samples)
    angles_rad = np.deg2rad(angles_deg)

    # Compute offsets
    ra = ra_center + r * np.cos(angles_rad)
    dec = dec_center + r * np.sin(angles_rad)

    # Build names
    names = [
        f"{quasar_name}_HSC_{fltr}_rotated_{angle:.2f}_degrees"
        for angle in angles_deg for fltr in fltrs
    ]

    return ra.tolist(), dec.tolist(), names


def table_writer(args):
    table, i, filepath_prefix, overwrite, tableformat, comment = args
    table.write(filepath_prefix + "download_sql" + ".txt",
                overwrite=overwrite, format=tableformat,
                comment=comment)

def write_downloader_files(
    df,
    step,
    filepath_prefix=None,
    write=True,
    overwrite=True,
    fltr=None,
    downloader_function=None
):
    n = len(df)
    chunk_args = [
        (df, start, min(start + step, n), fltr)
        for start in range(0, n, step)
    ]

    with mp.Pool() as pool:
        tables = list(
            tqdm(
                pool.imap(downloader_function, chunk_args),
                total=len(chunk_args),
                desc="Generating tables",
            )
        )

    if write:
        for i, table in enumerate(tqdm(tables, desc="Writing tables")):
            table_writer(
                (
                    table,
                    i,
                    filepath_prefix,
                    overwrite,
                    "ascii.commented_header",
                    "#?",
                )
            )
    else:
        for table in tables:
            print(table)

    return tables


def estimate_background(image: np.ndarray):
    data = image.astype(np.float32)
    bkg = sep.Background(data)

    return bkg.globalback, bkg.globalrms


def smart_combine(image1: np.ndarray, image2: np.ndarray):
    data1 = image1.astype(np.float32)
    data2 = image2.astype(np.float32)

    # Estimate the background level and noise of both images
    bkg_level1, bkg_rms1 = estimate_background(data1)
    bkg_level2, bkg_rms2 = estimate_background(data2)

    if bkg_rms1 == 0 or bkg_rms2 == 0:
        print("Background RMS of one of the images is zero, cannot scale the image.")
        return None

    # Subtract background levels
    data1 -= bkg_level1
    data2 -= bkg_level2

    # Combine the images
    combined_image = data1 + data2

    # Adjust the combined image to match the background level and noise of the first image
    combined_image = combined_image / 2  # Scale down to original noise level
    combined_image = combined_image - np.mean(combined_image) + bkg_level1
    return combined_image