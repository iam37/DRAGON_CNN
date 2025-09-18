# Standard library
import csv
import glob
import logging
import os
import random
import re
import shutil
import sys
import time
import warnings
from os import listdir
from os.path import exists, isfile, join
from pathlib import Path
from shutil import copy2
from time import sleep
from typing import Iterable

# Third-party general
import multiprocess as mp
import numpy as np
import pandas as pd
import scipy
import sep
import tqdm
from tqdm import tqdm

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# Astropy core
import astropy
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, angular_separation
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Moffat2D
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

# Photutils
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
)
from photutils.datasets import make_noise_image
from photutils.detection import DAOStarFinder
from photutils.psf import MoffatPSF, make_psf_model

# Other scientific/astro
import emcee
import lmfit
from schwimmbad import MultiPool


def crop_center(img, cropx, cropy):
    # Function from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, ...]


def _first_usable_hdu(data_hdul, min_shape):
    """Return the first 2D HDU with shape >= min_shape, else None."""
    H, W = min_shape
    for hdu in data_hdul:
        arr = getattr(hdu, "data", None)
        if arr is None:
            continue
        if arr.ndim == 2 and arr.shape[0] >= H and arr.shape[1] >= W:
            return arr
    return None


def sdss_to_dataframe(sdss_table, get_base_name):
    """
    Convert SDSS FITS table -> DataFrame with a 'base_name' key for joins.

    Expected columns (by index):
      RA: field(1), Dec: field(2), Name parts: field(6)+'_'+field(0), Redshift: field(28)
    """
    ra = sdss_table.field(1).byteswap().newbyteorder()
    dec = sdss_table.field(2).byteswap().newbyteorder()
    z = sdss_table.field(28).byteswap().newbyteorder()
    names = (sdss_table.field(6) + "_" + sdss_table.field(0)).byteswap().newbyteorder()

    df = pd.DataFrame({
        "Quasar name": names,
        "RA": ra,
        "Dec": dec,
        "Z": z
    })
    df["base_name"] = df["Quasar name"].apply(get_base_name)
    return df[["base_name", "Quasar name", "RA", "Dec", "Z"]]


get_base_name = lambda path: os.path.basename(path)


class InferenceResults:
    def __init__(self, df: pd.DataFrame, label_keys, data_dir, gothic_df: pd.DataFrame = None):
        label_dict = label_keys.set_index('key')['value'].to_dict()
        self.label_dict = {v: k for k, v in label_dict.items()}
        self.df = df
        self.data_dir = data_dir
        self.gothic_df = gothic_df

    def calculate_angular_separation(self, ra1, dec1, ra2, dec2, degrees: bool = True):
        """Great-circle angular separation via the haversine formula."""
        if degrees:
            ra1 = np.deg2rad(ra1)
            dec1 = np.deg2rad(dec1)
            ra2 = np.deg2rad(ra2)
            dec2 = np.deg2rad(dec2)

        d_ra = ra2 - ra1
        d_dec = dec2 - dec1

        # Haversine
        sin_ddec = np.sin(d_dec * 0.5)
        sin_dra = np.sin(d_ra * 0.5)
        a = sin_ddec ** 2 + np.cos(dec1) * np.cos(dec2) * sin_dra ** 2

        # Numerical guard: clip into [0, 1] before asin
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))

        return np.rad2deg(c) if degrees else c

    def double_moffat_model(self, theta: Iterable[float],
                            x: np.ndarray,
                            y: np.ndarray,
                            flatten: bool = True) -> np.ndarray:
        """Sum of two 2D Moffat profiles plus a bilinear background plane."""
        (x01, y01, A1, alpha1, gamma1,
         x02, y02, A2, alpha2, gamma2,
         pl_0, pl_x, pl_y, pl_xy) = theta

        # Two Moffat components
        m1 = Moffat2D(amplitude=A1, x_0=x01, y_0=y01, alpha=alpha1, gamma=gamma1)
        m2 = Moffat2D(amplitude=A2, x_0=x02, y_0=y02, alpha=alpha2, gamma=gamma2)

        # Background plane (bilinear)
        background = pl_0 + pl_x * x + pl_y * y + pl_xy * (x * y)

        model = m1(x, y) + m2(x, y) + background
        return model.ravel() if flatten else model

    def calculate_distance(self, QSO, quasar_name, wcs, pixel_scale=0.168, plot=False):
        """
        Find two bright components in an image, fit a double Moffat PSF, compute their separation,
        convert fitted centroids to RA/Dec, optionally reconcile with GOTHIC results, and return both
        pixel and sky info (including the SECONDARY RA/Dec).

        Returns
        -------
        sep_arcsec : float
            Final chosen separation in arcseconds (GOTHIC-preferred if plausible).
        (x1,y1) : tuple
            Fitted primary centroid (pixels).
        (x2,y2) : tuple
            Fitted secondary centroid (pixels).
        (ra1,dec1) : tuple
            Fitted primary sky position (degrees).
        (ra2,dec2) : tuple
            Fitted secondary sky position (degrees).
        """

        # ----------------------------
        # 0) Small local helpers
        # ----------------------------

        def prep_image(img):
            """Endian-safe, finite-only image + background subtraction for SEP."""
            # 1) Guarantee a NumPy array
            arr = np.array(img, copy=False)

            # 3) Ensure float dtype SEP likes (float32 is typical)
            arr = arr.astype(np.float32, copy=False)

            # 4) Replace NaN/±Inf with finite values before background modeling
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            # 5) Build background model (defaults are fine; tune bw/bh if needed)
            bkg = sep.Background(arr)

            # 6) Subtract the *background map*, not the Background object
            sub = arr - bkg.back()

            # 7) Sanity check
            if not np.any(np.isfinite(sub)):
                raise ValueError("No finite values after background subtraction.")

            return arr, sub, bkg

        def mesh(shape):
            """x,y grids for model evaluation."""
            h, w = shape
            y, x = np.mgrid[:h, :w]
            return x, y

        def detect_sources(img_sub, bkg, sigma=2.5):
            """Detect sources on the background-subtracted image."""
            thresh = sigma * bkg.globalrms
            cat = sep.extract(img_sub, thresh)
            if len(cat) == 0:
                raise ValueError("No sources detected.")
            order = np.argsort(cat['flux'])[::-1]
            return cat[order]

        def pick_two_peaks(cat):
            """Choose primary (brightest) and a valid secondary."""
            x, y = cat['x'], cat['y']
            if len(cat) < 2:
                return (x[0], y[0]), (x[0], y[0])  # single detection fallback
            d01 = np.hypot(x[0] - x[1], y[0] - y[1])
            j = 2 if (d01 >= 35.71 and len(cat) >= 3) else 1
            return (x[0], y[0]), (x[j], y[j])

        def init_double_moffat(img, c1, c2, alpha=5.1, beta=3.2):
            """Two Moffat PSFs seeded with image max flux."""
            flux0 = float(np.max(img))
            m1 = MoffatPSF(flux=flux0, x_0=c1[0], y_0=c1[1], alpha=alpha, beta=beta)
            m2 = MoffatPSF(flux=flux0, x_0=c2[0], y_0=c2[1], alpha=alpha, beta=beta)
            return m1 + m2

        def fit_model(model, x, y, data):
            """Levenberg–Marquardt fit; return fitted model."""
            fitter = fitting.LevMarLSQFitter()
            fitted = fitter(model, x, y, data)  # robust and simple
            return fitted

        def pix2world_xy(wcs_obj, xy):
            """(x,y) pixels -> (ra,dec) degrees; origin=1."""
            pts = np.array([xy], dtype=float)
            ra, dec = wcs_obj.all_pix2world(pts, 1)[0]
            return float(ra), float(dec)

        def sky_sep_arcsec(ra1, dec1, ra2, dec2):
            """Great-circle separation (arcsec) using SkyCoord."""
            c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
            c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
            return float(c1.separation(c2).to(u.arcsec).value)

        def pix_sep_arcsec(p1, p2, scale):
            """Pixel distance (arcsec) via scale."""
            return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]) * scale)

        def parse_gothic_peaks(peaks_str):
            """Parse GOTHIC 'u-peaks' column into two (x,y) tuples."""
            chunks = peaks_str[1:-1].split(")'")
            out = []
            for ch in chunks:
                pairs = re.findall(r"\((\d+),\s*(\d+)\)", ch)
                out.append([(int(x), int(y)) for x, y in pairs])
            if not out or len(out[0]) < 2:
                raise ValueError("Failed to parse GOTHIC peaks.")
            # Original convention: stored as (y,x); convert to (x,y)
            (y1, x1), (y2, x2) = out[0]
            return (x1, y1), (x2, y2)

        def quick_plot(img, pts, title):
            """Optional visualization."""
            if not plot:
                return
            vmin, vmax = np.percentile(img, [1, 99])
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, s=30, c="white", marker="x", label="Centroids")
            plt.title(title)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ----------------------------
        # 1) Prepare image and detect
        # ----------------------------
        img, img_sub, bkg = prep_image(QSO)
        xg, yg = mesh(img.shape)
        cat = detect_sources(img_sub, bkg, sigma=2.5)
        seed1, seed2 = pick_two_peaks(cat)

        # ----------------------------
        # 2) Fit double Moffat PSF
        # ----------------------------
        model0 = init_double_moffat(img, seed1, seed2)
        fitted = fit_model(model0, xg, yg, img)

        x1, y1 = fitted[0].x_0.value, fitted[0].y_0.value
        x2, y2 = fitted[1].x_0.value, fitted[1].y_0.value
        ra1, dec1 = pix2world_xy(wcs, (x1, y1))
        ra2, dec2 = pix2world_xy(wcs, (x2, y2))

        # Two ways to compute separation; prefer sky on success
        sep_fit_pix = pix_sep_arcsec((x1, y1), (x2, y2), pixel_scale)
        sep_fit_sky = sky_sep_arcsec(ra1, dec1, ra2, dec2)
        sep_fit = sep_fit_sky if np.isfinite(sep_fit_sky) else sep_fit_pix

        quick_plot(img, [(x1, y1), (x2, y2)], f"{quasar_name} — Fitted Moffat Centroids")

        # ----------------------------
        # 3) Optional GOTHIC reconciliation
        # ----------------------------
        if self.gothic_df is None:
            return sep_fit, (x1, y1), (x2, y2), (ra1, dec1), (ra2, dec2)

        print("Attempting Gothic")
        basename = os.path.basename(quasar_name)[:-5]
        row = self.gothic_df[self.gothic_df["objid"] == basename]
        if row.empty:
            return sep_fit, (x1, y1), (x2, y2), (ra1, dec1), (ra2, dec2)

        utypes = set(row["u-type"].values.tolist())
        if any(tag in utypes for tag in ("NO_PEAK", "SINGLE", "ERROR")):
            return sep_fit, (x1, y1), (x2, y2), (ra1, dec1), (ra2, dec2)

        # Parse peaks → world coords → separation
        try:
            peaks_str = row["u-peaks"].values[0]
            (gx1, gy1), (gx2, gy2) = parse_gothic_peaks(peaks_str)
            gra1, gdec1 = pix2world_xy(wcs, (gx1, gy1))
            gra2, gdec2 = pix2world_xy(wcs, (gx2, gy2))
            sep_gothic = sky_sep_arcsec(gra1, gdec1, gra2, gdec2)
            quick_plot(img, [(gx1, gy1), (gx2, gy2)], f"{quasar_name} — GOTHIC Centroids")
        except Exception:
            # If GOTHIC parsing fails, stick to fit
            return sep_fit, (x1, y1), (x2, y2), (ra1, dec1), (ra2, dec2)

        # ----------------------------
        # 4) Final choice policy
        # ----------------------------
        IMPOSSIBLE = 8.0  # arcsec guardrail
        if sep_gothic >= IMPOSSIBLE and sep_fit < IMPOSSIBLE:
            return sep_fit, (x1, y1), (x2, y2), (ra1, dec1), (ra2, dec2)
        if sep_fit >= IMPOSSIBLE and sep_gothic < IMPOSSIBLE:
            return sep_gothic, (gx1, gy1), (gx2, gy2), (gra1, gdec1), (gra2, gdec2)

        # Otherwise prefer GOTHIC when both are reasonable
        return sep_gothic, (gx1, gy1), (gx2, gy2), (gra1, gdec1), (gra2, gdec2)

    def build_candidate_rows(self, image_paths, sdss_df, secondary_lookup=None):
        """
        Build list[dict] rows for CSV.
        - `secondary_lookup(fname) -> (ra2, dec2)` (optional)
        - Else tries self.secondary_sky_coords[fname]
        """
        rows = []
        for img in image_paths:
            p = Path(img)
            base = p.stem
            fname = p.name

            match = sdss_df.loc[sdss_df["base_name"] == base]
            if match.empty:
                continue

            z = match["Z"].values[0]
            ra1 = match["RA"].values[0]  # primary RA from SDSS
            dec1 = match["Dec"].values[0]  # primary Dec from SDSS

            # existing maps you keep:
            dist = self.distance_map.get(fname, np.nan)
            cen_mag = self.central_instrumental_mags.get(fname, np.nan)
            sec_mag = self.secondary_instrumental_mags.get(fname, np.nan)
            x_unc = self.x_errors.get(fname, np.nan)
            ang_err = self.calculate_ang_sep_error(ra1, dec1, x_unc) if np.isfinite(x_unc) else np.nan

            # --- NEW: get secondary sky coords (ra2, dec2) ---
            ra2 = dec2 = np.nan
            if secondary_lookup is not None:
                try:
                    ra2, dec2 = secondary_lookup(fname)
                except Exception:
                    pass
            elif hasattr(self, "secondary_sky_coords"):
                ra2, dec2 = self.secondary_sky_coords.get(fname, (np.nan, np.nan))

            rows.append({
                "Candidate name": base,
                "RA": ra1,
                "Dec": dec1,
                "z": z,
                "secondary_RA": ra2,  # <— added
                "secondary_Dec": dec2,  # <— added
                "angular_separation": dist,
                "angular_separation_error": ang_err,
                "central_mag": cen_mag,
                "secondary_mag": sec_mag,
            })
        return rows

    def write_candidate_csv(self, rows, out_csv_path):
        cols = [
            "Candidate name", "RA", "Dec", "z",
            "secondary_RA", "secondary_Dec",  # <— added
            "angular_separation", "angular_separation_error",
            "central_mag", "secondary_mag",
        ]
        df = pd.DataFrame(rows, columns=cols)
        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
        return df

    def plot_redshift_separations(self, candidate_df=None, save=True):
        """
        Plot Angular Separation vs. Redshift. If candidate_df is None, reads
        '{self.dual_AGN_filepath}/complete_candidate_info.csv'.
        """
        if candidate_df is None:
            csv_path = Path(self.data_dir) / "pred_dual_AGN" / "complete_candidate_info.csv"
            candidate_df = pd.read_csv(csv_path)

        # Mask invalid redshifts if the sentinel is -1.0
        mask = candidate_df["z"].values != -1.0
        redshifts = candidate_df.loc[mask, "z"].values
        separations = candidate_df.loc[mask, "angular_separation"].values

        plt.figure(figsize=(7, 5))
        plt.scatter(redshifts, separations, s=10, label="Dual AGN Candidates")
        plt.xlabel("Redshift z")
        plt.ylabel("Angular Separation [arcsec]")
        plt.title("Redshift vs. Angular Separation (HSC Dual AGN Candidates)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save:
            out_png = Path(self.data_dir) / "pred_dual_AGN" / "redshift_angular_separation.png"
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.show()

    def load_fits_images(self, df, data_dir, target_shape=(94, 94)):
        """
        Returns:
          training_images, image_files, true_labels, pred_labels, pred_confidences, mismatched_images
        """
        training_images = []
        image_files = []
        true_labels = []
        pred_labels = []
        pred_confidences = []
        mismatched_images = []

        data_dir = Path(data_dir)

        # Pull these once to avoid repeated global lookups
        _label_dict = self.label_dict if hasattr(self, "label_dict") else label_dict  # keep legacy behavior
        _crop = self.cropND if hasattr(self, "cropND") else cropND  # keep legacy behavior

        for i, row in tqdm(enumerate(df.itertuples(index=False)), total=len(df)):
            # Row fields (allow missing columns gracefully)
            file_name = getattr(row, "file_name")
            image_class = getattr(row, "classes", -1)
            pred_label_raw = getattr(row, "predicted_labels")
            pred_conf = getattr(row, "predicted_confidence")

            img_path = data_dir / file_name

            # Open once; pick first usable HDU
            try:
                with fits.open(img_path, memmap=False) as hdul:
                    img = _first_usable_hdu(hdul, target_shape)
                    if img is None:
                        # No usable HDU; skip
                        continue

                    # endian/NaN safe
                    img = np.nan_to_num(img.byteswap().newbyteorder(), nan=0.0, posinf=0.0, neginf=0.0)

                    # Crop to target
                    img = _crop(img, target_shape)

                    training_images.append(img)
                    image_files.append(str(img_path))
                    true_labels.append(image_class)
                    pred_labels.append(_label_dict[pred_label_raw])
                    pred_confidences.append(pred_conf)

                    # Mismatch bookkeeping
                    if image_class != _label_dict[pred_label_raw]:
                        mismatched_images.append(i)
            except Exception as e:
                logging.exception(f"Error reading {img_path}: {e}")
                continue

        return training_images, image_files, true_labels, pred_labels, pred_confidences, mismatched_images

    def sort_classifications(self):
        # Known confirmations
        CANDIDATES = {"220906_91", "221115_06", "222057_44", "220718_43", "220811_56", "silverman"}

        base = Path(self.data_dir)

        # Centralized destinations per voted class
        # 0=rubbish, 1=empty, 2=single, 3=offset, 4=merger, 5=dual
        dest_by_class = {
            0: base / "pred_rubbish",
            1: base / "pred_empty_space",
            2: base / "pred_single_AGN",
            3: base / "pred_offset_AGN",
            4: base / "pred_mergers",
            5: base / "pred_dual_AGN",
        }

        # Also a special folder for previously confirmed candidates
        confirmed_dir = base / "prev_confirmed_candidates"

        # Ensure all destinations exist
        for p in [confirmed_dir, *dest_by_class.values()]:
            p.mkdir(parents=True, exist_ok=True)

        def candidate_key(stem: str) -> str:
            """
            From a filename stem like 'HSC_220906_91_g.fits' produce candidate tag.
            Rule preserved from your code:
              - if there are >3 underscore parts: use parts[1] + '_' + parts[2]
              - else: use parts[0]
            """
            parts = stem.split("_")
            return "_".join([parts[1], parts[2]]) if len(parts) > 3 else parts[0]

        # Iterate efficiently
        for row in tqdm(self.df.itertuples(index=False), total=len(self.df)):
            file_name = getattr(row, "file_name")
            voted_class = getattr(row, "voted_class")
            src = base / file_name

            if not src.exists():
                print(f"Missing file: {src}")
                continue

            stem = Path(file_name).stem
            cand = candidate_key(stem)

            # Confirmed bucket overrides all else
            if cand in CANDIDATES:
                copy2(src, confirmed_dir)
                continue

            # Route by class (guard unknowns)
            dst_dir = dest_by_class.get(voted_class)
            if dst_dir is None:
                print(f"Unknown voted_class={voted_class} for {src.name}; skipping.")
                continue

            copy2(src, dst_dir)

        # Simple summary
        print(self.df["voted_class"].value_counts())

    def plot_voted_separations(
            self,
            congress_df: pd.DataFrame,
            confidence_threshold: float | None = None,
            candidate_class: int = 5,
            unanimous: bool = True,
            *,
            sep_bins: tuple[float, float, int] | np.ndarray = (0.0, 7.0, 50),
            conf_bins: tuple[float, float, int] | np.ndarray = (0.0, 1.0, 50),
            figsize: tuple[int, int] = (10, 4),
            save: bool = True,
            outfile: str | Path = "voter_separation_confidence_plot.png",
            dpi: int = 300,
    ) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
        """
        Filter votes, compute angular separations via `self.calculate_distance`, and plot histograms
        of separations and weighted confidences.

        Returns:
            (fig, axes, results_df)
        """
        # ---------- Validate inputs ----------
        required = {"voted_class", "num_voters", "total_voters", "weighted_confidence", "file_name"}
        missing = required - set(congress_df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        # ---------- Configure bins (accept either (start, stop, num) or explicit edges) ----------
        def _bins(b):
            if isinstance(b, (tuple, list)) and len(b) == 3:
                start, stop, num = b
                return np.linspace(float(start), float(stop), int(num))
            return np.asarray(b, dtype=float)

        sep_bins_arr = _bins(sep_bins)
        conf_bins_arr = _bins(conf_bins)

        # ---------- Helpers to normalize outputs ----------
        def _unpack_radec(radec) -> tuple[float, float]:
            # Accepts SkyCoord or (ra, dec) in degrees
            try:
                # SkyCoord-like
                if hasattr(radec, "ra") and hasattr(radec, "dec"):
                    # .ra/.dec can be Angle objects
                    ra_deg = float(getattr(radec.ra, "deg", radec.ra))
                    dec_deg = float(getattr(radec.dec, "deg", radec.dec))
                    return ra_deg, dec_deg
            except Exception:
                pass
            # Tuple/list/array fallback (assumed degrees)
            if isinstance(radec, (tuple, list, np.ndarray)) and len(radec) >= 2:
                return float(radec[0]), float(radec[1])
            raise TypeError(f"Unrecognized RA/Dec format: {type(radec)}")

        # ---------- Iterate rows, filter, measure ----------
        data_dir = Path(self.data_dir)
        distances_arcsec: list[float] = []
        names: list[str] = []
        weighted_confidences: list[float] = []

        # NEW: coordinate accumulators
        center_ras: list[float] = []
        center_decs: list[float] = []
        sec_ras: list[float] = []
        sec_decs: list[float] = []

        for row in tqdm(congress_df.itertuples(index=False), total=len(congress_df)):
            # Filters (short-circuit)
            if row.voted_class != candidate_class:
                continue
            if unanimous and (row.num_voters != row.total_voters):
                continue
            if (confidence_threshold is not None) and (row.weighted_confidence < confidence_threshold):
                continue

            image_path = data_dir / row.file_name
            if not image_path.exists():
                continue

            # Open FITS and build a WCS (prefer HDU 1 if available else primary)
            try:
                with fits.open(image_path, memmap=False) as hdul:
                    try:
                        hdu = _first_usable_hdu(hdul, min_shape=(1, 1)) if len(hdul) > 1 else hdul[0]
                    except NameError:
                        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                    img = hdu.data if hdu.data is not None else fits.getdata(image_path, memmap=False)
                    try:
                        wcs = WCS(hdu.header)
                    except Exception:
                        wcs = WCS(hdul[1].header) if len(hdul) > 1 else WCS(hdul[0].header)
            except Exception as e:
                logging.exception(f"Error reading {image_path}: {e}")
                continue

            # Use your existing distance routine
            try:
                sep_arcsec, center_xy, secondary_xy, center_radec, secondary_radec = self.calculate_distance(
                    QSO=img, quasar_name=row.file_name, wcs=wcs, plot=True
                )
            except Exception as e:
                logging.exception(f"Error calculating {image_path}: {e}")
                continue

            # Collect scalars
            distances_arcsec.append(float(sep_arcsec))
            names.append(row.file_name)
            weighted_confidences.append(float(row.weighted_confidence))

            # Unpack and collect coordinates
            cra, cdec = _unpack_radec(center_radec)
            sra, sdec = _unpack_radec(secondary_radec)

            center_ras.append(cra)
            center_decs.append(cdec)
            sec_ras.append(sra)
            sec_decs.append(sdec)

        # Back-compat field
        self.weighted_confidences = weighted_confidences

        print(
            f"Number of dual AGN candidates "
            f"{'unanimously ' if unanimous else ''}voted with confidence "
            f"{'over ' + str(confidence_threshold) if confidence_threshold is not None else '(no threshold)'}: "
            f"{len(distances_arcsec)}"
        )

        # ---------- Results table for downstream use ----------
        results_df = pd.DataFrame(
            {
                "file_name": names,
                "distance_arcsec": distances_arcsec,
                "weighted_confidence": weighted_confidences,
                "center_ra_deg": center_ras,
                "center_dec_deg": center_decs,
                "secondary_ra_deg": sec_ras,
                "secondary_dec_deg": sec_decs,
            }
        )

        # ---------- Plot ----------
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        n_sep, _, _ = axes[0].hist(
            distances_arcsec,
            bins=sep_bins_arr,
            label="Angular Separations",
            color="midnightblue",
        )
        axes[0].set_xlabel("Angular Separation [arcsec]", fontsize=8)
        if len(sep_bins_arr) > 1:
            axes[0].set_xlim(sep_bins_arr.min(), sep_bins_arr.max())
        axes[0].set_ylabel("Number of Dual AGN Candidates", fontsize=8)
        axes[0].set_title("Angular Separations", fontsize=10)
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)

        n_conf, _, _ = axes[1].hist(
            weighted_confidences,
            bins=conf_bins_arr,
            label="Weighted Confidences",
            color="lightsalmon",
        )
        axes[1].set_xlabel("Weighted Confidence", fontsize=8)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_ylabel("Number of Dual AGN Candidates", fontsize=8)
        axes[1].set_title("Weighted Confidences", fontsize=10)
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.3)

        if save:
            out_path = Path(outfile)
            if not out_path.is_absolute():
                out_path = Path(self.data_dir) / out_path
            try:
                fig.savefig(out_path, dpi=dpi)
            except Exception:
                pass  # Non-fatal

        plt.show()
        return fig, axes, results_df

    def plot_magnitudes(
            self,
            congress_df: pd.DataFrame,
            confidence_threshold=None,
            candidate_class: int = 5,
            unanimous: bool = True,
            *,
            crop_size: int = 94,
            bins_mag: int = 30,
            bins_inst: int = 25,
            bins_flux: int = 30,
            figsize=(10, 4),
            save: bool = True,
            outfile: str | Path = "binned_magnitude_plots.png",
            dpi: int = 300,
    ):
        """
        Build magnitude/flux-ratio histograms for selected candidates and plot them.

        Returns:
            fig, axes, results_df
              - results_df columns: ['file_name','inst_mag_center','inst_mag_secondary','flux_ratio','mag_difference']
        """
        # ---- sanity checks ----
        required = {"voted_class", "num_voters", "total_voters", "weighted_confidence", "file_name"}
        missing = required - set(congress_df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        # ---- collectors ----
        candidate_mag_differences: list[float] = []
        flux_ratios: list[float] = []
        names: list[str] = []
        inst_centers: list[float] = []
        inst_seconds: list[float] = []

        self.central_instrumental_mags = {}
        self.secondary_instrumental_mags = {}

        data_dir = Path(self.data_dir)

        # ---- iterate once; unify filters ----
        for row in tqdm(congress_df.itertuples(index=True, name="Row"), total=len(congress_df)):
            # class filter
            if row.voted_class != candidate_class:
                continue
            # unanimity filter
            if unanimous and (row.num_voters != row.total_voters):
                continue
            # confidence filter
            if (confidence_threshold is not None) and (row.weighted_confidence < confidence_threshold):
                continue

            image_path = data_dir / row.file_name
            if not image_path.exists():
                # optional: print(f"Could not find {image_path}")
                continue

            # ---- FITS I/O (context-managed) ----
            try:
                with fits.open(image_path, memmap=False) as hdul:
                    # image HDU: prefer [0] data; some stacks use [1] for image + WCS
                    hdu_img = None
                    # choose the first HDU with non-None data
                    for h in hdul:
                        if getattr(h, "data", None) is not None:
                            hdu_img = h
                            break
                    if hdu_img is None:
                        # fallback: try getdata
                        img = fits.getdata(image_path, memmap=False)
                    else:
                        img = hdu_img.data

                    # crop for photometry (uses your helper)
                    img = crop_center(img, crop_size, crop_size)

                    # FLUXMAG0 (robust cast)
                    fluxmag_0_raw = hdul[0].header.get("FLUXMAG0", None)
                    if fluxmag_0_raw is None:
                        # if not present, skip this file
                        continue
                    try:
                        fluxmag_0 = float(str(fluxmag_0_raw).strip())
                    except Exception:
                        # header value malformed; skip
                        continue

                    # WCS: prefer extension 1 if present, else primary
                    try:
                        header_wcs = hdul[1].header if len(hdul) > 1 else hdul[0].header
                        wcs = WCS(header_wcs)
                    except Exception:
                        # last-ditch: WCS from primary
                        wcs = WCS(hdul[0].header)
            except Exception:
                # unreadable FITS -> skip
                continue

            # ---- compute separation and magnitudes ----
            try:
                # your calculate_distance may return extra fields; accept and ignore extras with unpacking
                out = self.calculate_distance(img, row.file_name, wcs)
                # handle either 3-tuple or longer tuples
                distance_arcsec = out[0]
                center_coords = out[1]
                secondary_coords = out[2]
            except Exception as e:
                logging.exception(f"Error in parsing distance: {e}")
                continue

            # Prefer the freshly computed separation over self.distances[idx]
            try:
                inst_mag_1, inst_mag_2, flux_ratio, mag_diff = self.calculate_magnitudes(
                    center_coords, secondary_coords, fluxmag_0, img, distance_arcsec
                )
            except Exception:
                continue

            # collect
            candidate_mag_differences.append(float(mag_diff))
            flux_ratios.append(float(flux_ratio))
            names.append(row.file_name)
            inst_centers.append(float(inst_mag_1))
            inst_seconds.append(float(inst_mag_2))
            self.central_instrumental_mags[row.file_name] = float(inst_mag_1)
            self.secondary_instrumental_mags[row.file_name] = float(inst_mag_2)

        print(
            f"Number of dual AGN candidates "
            f"{'unanimously ' if unanimous else ''}voted with confidence "
            f"{'over ' + str(confidence_threshold) if confidence_threshold is not None else '(no threshold)'}: "
            f"{len(candidate_mag_differences)}"
        )

        # ---- results table ----
        results_df = pd.DataFrame(
            {
                "file_name": names,
                "inst_mag_center": inst_centers,
                "inst_mag_secondary": inst_seconds,
                "flux_ratio": flux_ratios,
                "mag_difference": candidate_mag_differences,
            }
        )

        # ---- plotting ----
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

        # (1) magnitude differences
        axes[0].hist(candidate_mag_differences, bins=bins_mag, histtype="step",
                     color="midnightblue", label="New Candidates")
        axes[0].set_xlabel("Pairwise Magnitude Difference [AB]", fontsize=8)
        axes[0].set_ylabel("Number of Dual AGN Candidates", fontsize=8)
        axes[0].set_title("Magnitude Differences", fontsize=10)
        axes[0].set_xlim(0, 5)
        axes[0].legend(loc=0)
        axes[0].grid(True, which="both", alpha=0.3)

        # pretty ticks
        def _pretty_ticks(ax, n=7):
            ticks = ax.get_xticks()
            if len(ticks) >= 2:
                new = np.linspace(ticks[0], ticks[-1], n)
                ax.set_xticks(new)
                ax.set_xticklabels([f"{t:.2f}" for t in new], rotation=45, ha="right")

        _pretty_ticks(axes[0], n=7)

        # (2) instrumental magnitudes
        axes[1].hist(inst_centers, bins=bins_inst, histtype="step",
                     color="midnightblue", label="Central quasars")
        axes[1].hist(inst_seconds, bins=bins_inst, histtype="step",
                     color="lightsalmon", label="Secondary objects")
        axes[1].set_xlabel("Instrumental Magnitude [AB]", fontsize=8)
        axes[1].set_ylabel("Number of Dual AGN Candidates", fontsize=8)
        axes[1].set_title("Central and Secondary Object Magnitudes", fontsize=10)
        axes[1].legend(loc=0)
        axes[1].grid(True, which="both", alpha=0.3)
        _pretty_ticks(axes[1], n=7)

        # (3) flux ratios (log bins)
        logbins = np.logspace(0, 4, bins_flux + 1)
        axes[2].hist(flux_ratios, bins=logbins, histtype="step",
                     color="firebrick", label="New Candidates")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("Flux Ratios [unitless]", fontsize=8)
        axes[2].set_ylabel("Number of Dual AGN Candidates", fontsize=8)
        axes[2].set_title("Pairwise Flux Ratios", fontsize=10)
        axes[2].legend(loc=0)
        axes[2].grid(True, which="both", alpha=0.3)
        axes[2].xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else "")
        )
        axes[2].xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))  # reduce clutter
        for tick in axes[2].get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

        # save/show
        if save:
            out = Path(outfile)
            if not out.is_absolute():
                out = data_dir / out
            try:
                fig.savefig(out, dpi=dpi)
            except Exception:
                pass

        plt.show()
        return fig, axes, results_df

    def export_and_plot_candidates(self, sdss_table):
        sdss_df = sdss_to_dataframe(sdss_table, get_base_name)
        images = glob.glob(str(Path(self.data_dir) / "pred_dual_AGN" / "*.fits"))

        # Option A: if you already populated self.secondary_sky_coords earlier:
        rows = self.build_candidate_rows(images, sdss_df)

        out_csv = Path(self.data_dir) / "pred_dual_AGN" / "complete_candidate_info.csv"
        candidate_df = self.write_candidate_csv(rows, out_csv)

        self.complete_candidate_df = candidate_df
        self.plot_redshift_separations(candidate_df=candidate_df, save=True)

    def log_prior(self, theta, xlim, ylim):
        """Uniform (hard) priors: amplitudes/scales positive, centers inside frame."""
        (x01, y01, A1, a1, g1,
         x02, y02, A2, a2, g2,
         pl_0, pl_x, pl_y, pl_xy) = theta

        xmin, xmax = xlim
        ymin, ymax = ylim

        # Physically sensible ranges
        if not (0.0 < A1 < 1e5 and 0.0 < A2 < 1e5): return -np.inf
        if not (0.0 < a1 < 20.0 and 0.0 < a2 < 20.0): return -np.inf
        if not (0.0 < g1 < 20.0 and 0.0 < g2 < 20.0): return -np.inf
        if not (xmin < x01 < xmax and xmin < x02 < xmax): return -np.inf
        if not (ymin < y01 < ymax and ymin < y02 < ymax): return -np.inf

        return 0.0  # flat within bounds

    def log_likelihood(self, theta, x, y, data, sigma):
        """Gaussian log-likelihood with normalization."""
        model = self.double_moffat_model(theta, x, y, flatten=True)

        d = np.ravel(data)
        s = np.ravel(sigma)
        # Guard against zero/negatives in sigma
        s = np.where(s > 0, s, np.median(s[s > 0]))

        r = (d - model) / s
        return -0.5 * (np.sum(r ** 2) + np.sum(np.log(2.0 * np.pi * s ** 2)))

    def log_probability(self, theta, x, y, data, sigma, xlim=None, ylim=None):
        """Posterior log-probability = prior + likelihood."""
        # Derive bounds once if not provided
        if xlim is None: xlim = (np.min(x), np.max(x))
        if ylim is None: ylim = (np.min(y), np.max(y))

        lp = self.log_prior(theta, xlim, ylim)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, data, sigma)


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'serif'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')