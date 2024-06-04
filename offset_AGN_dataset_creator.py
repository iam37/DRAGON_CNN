import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
import astropy.units as u 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.utils.data import get_pkg_data_filename
from astropy.cosmology import WMAP9 as cosmo
from photutils import DAOStarFinder, CircularAperture, CircularAnnulus, aperture_photometry, RectangularAperture, RectangularAnnulus
from astropy.modeling.models import Gaussian2D, Sersic2D
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.nddata import CCDData
from astropy.modeling.models import Moffat2D
from astropy.convolution import convolve
from astropy.convolution import Moffat2DKernel
from astropy.stats import gaussian_sigma_to_fwhm
import glob
from tqdm import tqdm
import os
from os.path import exists

class DatasetCreator:
    def __init__(self):
        pass
    def crop_center(self, img, cropx, cropy):
    
        #Function from  
        #https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
        
        y, x, *_ = img.shape
        startx = x // 2 - (cropx // 2)
        #print(startx)
        starty = y // 2 - (cropy // 2) 
        #print(starty)
        return img[starty:starty + cropy, startx:startx + cropx, ...]
    def make_random_galaxy_table(self, number, params, seed):
        """
        Generate a table of random galaxy parameters.
        """
        np.random.seed(seed)

        table = Table()
        #print(params)
        for param_name, param_range in params.items():
            param_values = np.random.uniform(param_range[0], param_range[1], size=number) #figure out why galaxies are only in the upper right corner
            table[param_name] = param_values
        #print(len(table))

        return table
    def make_galaxy_image(self, shape, galaxies):
        """
        Generate an image with galaxies using the provided parameters.
        """
        center_x = 30
        center_y = 30 #make sure to later change the coordinates for the center of images when I update this code for 
        #multiple sizes available.
        exclusion_radius = 5
        num_galaxies = len(galaxies)
        image = np.zeros(shape)
        side_length = int(np.ceil(np.sqrt(num_galaxies)))
        num_rows = min(side_length, shape[0])
        num_cols = min(side_length, shape[1])
        # Calculate the spacing between galaxies
        if num_rows != 0 and num_cols != 0:
            row_spacing = shape[0] / num_rows
            col_spacing = shape[1] / num_cols
            for i, galaxy in enumerate(galaxies):
                amplitude = galaxy['amplitude']
                r_eff = galaxy['r_eff']
                n = galaxy['n']
                row = int(i/num_cols)
                col = i%num_cols
                y_0 = (row + 0.5)*row_spacing
                x_0 = (col + 0.5)*col_spacing
                if np.sqrt((x_0 - center_x)**2 + (y_0 - center_y)**2) <= exclusion_radius:
                    return np.zeros(shape)
                #x_0 = galaxy['x_0']
                #y_0 = galaxy['y_0']
                phi = galaxy['phi']
                ellip = galaxy['ellipticity']

                model = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=x_0, y_0=y_0, ellip=ellip, theta = phi)
                galaxy_image = model(*np.indices(shape))
                img = image + galaxy_image
                return img if num_galaxies != 0 else np.zeros(shape)
        else:
            pass
    def addBackgroundGalaxies(self, image, number, flux, gain=1): #Adapted from Astropy.io: https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-03-Construction-of-an-artificial-but-realistic-image.html#add-some-stars
        """
        Add some background galaxies to make images less uniform and hopefully prevent the model from overfitting.
        """
        if number >= 0:
            x_stddev = np.random.uniform(1.5, 3.0)
            #print(x_stddev)
            y_stddev = np.random.uniform(1.5, 3.0)
            flux_range = [flux/20, flux]
            amplitude_range = flux_range
            r_eff_range = [1.0, 3.5]  # Effective radius range
            n_range = [0.5, 4.0]  # Sersic index range
            phi_range = [0, np.pi]  # Rotation angle range
            ellipticity_range = [1.0, 3.5]  # Ellipticity range
            y_max, x_max = image.shape
            xmean_range_lower = [0.2 * x_max, 0.35 * x_max]
            xmean_range_upper = [0.65*x_max, 0.8*x_max]
            ymean_range_lower = [0.2 * y_max, 0.35 * y_max]
            ymean_range_upper = [0.65*y_max, 0.8*y_max]
            params_index = np.random.randint(0, high = 1, size = 1)
            if params_index == 1:
                params_lower = dict([('amplitude', flux_range),
                            ('r_eff', r_eff_range),
                            ('n', n_range),
                            ('x_0', xmean_range_lower),
                            ('y_0', ymean_range_lower),
                            ('phi', phi_range),
                            ('ellipticity', ellipticity_range)])
                seed = np.random.randint(0, 10000)

                sources = self.make_random_galaxy_table(number, params_lower, seed)

                galaxy_im = self.make_galaxy_image(image.shape, sources)

                return galaxy_im
            elif params_index == 0:
                params_upper = dict([('amplitude', flux_range),
                            ('r_eff', r_eff_range),
                            ('n', n_range),
                            ('x_0', xmean_range_upper),
                            ('y_0', ymean_range_upper),
                            ('phi', phi_range),
                            ('ellipticity', ellipticity_range)])
                seed = np.random.randint(0, 10000)

                sources = self.make_random_galaxy_table(number, params_upper, seed)

                galaxy_im = self.make_galaxy_image(image.shape, sources)

                return galaxy_im
        else: 
            pass
    def extract_single_galaxies(self, galaxy_filepath):
        self.singleton_images = []
        for image in glob.glob(galaxy_filepath + "*.fits"):
            with fits.open(image, memmap = False) as hdul:
                img = hdul[0].data
                img = self.crop_center(img, 60, 60)
                self.singleton_images.append(img)
        self.singleton_names = [(s[:s.rfind("\\")], s[s.rfind("\\") + 1:-5]) if s.rfind("\\") != -1 else (s, "") for s in glob.glob(galaxy_filepath + "*.fits")]
    def extract_rotated_AGN(self, rotated_AGN_filepath_prefix):
        asec_separations = ["2.0","1.9", "1.8", "1.7", "1.6", "1.4", "1.3", "1.2", "1.1","1.0", "0.8", "0.7", "0.6", "0.5"]
        self.rotated_AGN = []
        for j in tqdm(asec_separations):
            for images in glob.glob(rotated_AGN_filepath_prefix + j + "_asec_separations/*.fits"):

                with fits.open(images, memmap = False) as hdu1:
                    comp_img = hdu1[0].data
                    comp_img = self.crop_center(comp_img, 60, 60)
                    comp_img = np.expand_dims(comp_img, axis = -1)

                self.rotated_AGN.append(comp_img)
            self.AGN_names = [(s[:s.rfind("\\")], s[s.rfind("\\") + 1:-5]) if s.rfind("\\") != -1 else (s, "") for s in glob.glob(rotated_AGN_filepath_prefix + j + "_asec_separations/*.fits")]

    def convolve_galaxy_AGN(self, galaxy_img, AGN_img):
        convolved_image = convolve_fft(AGN_img, galaxy_img)
        return convolved_image 
    def create_convolution(self, fits_filepath = "offset_AGN_images/"):
        if not exists(fits_filepath):
            os.makedirs(fits_filepath)
        for ii, galaxy_img in tqdm(enumerate(self.singleton_images)):
            for j, AGN_img in enumerate(self.rotated_AGN):
                convolved_image = self.convolve_galaxy_AGN(galaxy_img, AGN_img)
                AGN_name = self.AGN_names[j]
                galaxy_name = self.singleton_names[ii]
                hdu = fits.PrimaryHDU(convolved_image)
                hdul = fits.HDUList([hdu])
                hdul.writeto(f"{fits_filepath}{galaxy_name}_with_AGN_{AGN_name}.fits" , overwrite=True)





