import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.io import fits
import os
def modified_load_fits(filepath, extension = 0, explore = False):
    #file_extension = filepath[filepath.index('.'):] #locates the file extension
    try:
        hdu = fits.open(filepath)
        header = hdu[extension].header
        data = hdu[extension].data
        if explore == True:
            hdu.info()
        return header, data, hdu
    except:
        assert isinstance(filepath, str), "Input must be formatted as a string"
        file_extension = filepath[filepath.index('.'):]
        assert (file_extension == '.fits') or (file_extension == '.fit') or (file_extension == '.Fits') or (file_extension == '.Fit'), "File must be in proper FITS format"
        assert os.path.isfile(filepath), f"Could not find file or directory {filepath}"
def modified_plot_image(data, figsize = (15,3), cmap = 'gray_r', scale = 0.5, wcs = None, grid = False, title = "Image of Astronomical Object", figsave_filepath = None, **kwargs):
    mu = np.mean(data)
    sigma = np.std(data)
    temp_lower = mu -scale*sigma
    temp_upper = mu + scale*sigma
    if wcs is not None:
        fig, ax = plt.subplots(1,1, figsize = figsize, subplot_kw = {'projection': wcs})
        if 'vmin' and 'vmax' in kwargs.keys():
            if grid:
                ax.coords.grid(color = 'gray', alpha = 0.5, linestyle = 'solid')
            ax.imshow(data, cmap = cmap, **kwargs)
        else:
            if grid:
                ax.coords.grid(color = 'gray', alpha = 0.5, linestyle = 'solid')
            ax.imshow(data, cmap = cmap, vmin = temp_lower, vmax = temp_upper)
        plt.title(title)
        ax.invert_yaxis()
        ax.set_xlabel("Right Ascension [hms]")
        ax.set_ylabel("Declination [degrees]")
        #ax.grid()
        if figsave_filepath != None:
            plt.savefig(str(figsave_filepath), dpi = 300)
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        if 'vmin' and 'vmax' in kwargs.keys():
            if grid:
                ax.coords.grid(color = 'gray', alpha = 0.5, linestyle = 'solid')
            ax.imshow(data, cmap = cmap, **kwargs)
            ax.invert_yaxis()
        else:
            if grid:
                ax.coords.grid(color = 'gray', alpha = 0.5, linestyle = 'solid')
            ax.imshow(data, cmap = cmap, vmin = temp_lower, vmax = temp_upper)
        plt.title(title)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("x coordinate [arbitary units]")
        ax.set_ylabel("y coordinate [arbitrary units]")
        #ax.grid()
        if figsave_filepath != None:
            plt.savefig(str(figsave_filepath), dpi = 300)
        plt.show()
    return fig, ax

def plot_dataset_sample(dataset, labels, title = "Randomly Selected AGN", AGN_type = "single AGN"):
    mask = np.where(labels == AGN_type)
    print(len(mask))
    AGN_images = dataset[mask]
    random_index = np.random.choice(len(AGN_images))
    random_image = AGN_images[random_index]
    modified_plot_image(random_image, title = title, vmin = np.percentile(random_image, 1), vmax = np.percentile(random_image, 99))

