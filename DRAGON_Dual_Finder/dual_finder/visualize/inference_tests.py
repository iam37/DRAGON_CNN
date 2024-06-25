import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle

from astropy.utils.data import get_pkg_data_filename
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import FlatLambdaCDM
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import sys
import scipy
from astropy.modeling import models, fitting

import lmfit
import logging
import time
import sep
import csv

from photutils.datasets import make_noise_image
import os
from os import listdir
from os.path import isfile, join
from os.path import exists
from time import sleep 

import random
import csv
import warnings
from schwimmbad import MultiPool
from astropy.table import Table
import multiprocess as mp
import tqdm
from tqdm import tqdm
import shutil

import keras
from tensorflow.keras import  models
from tensorflow.keras.callbacks import EarlyStopping

def bubbleSort(array):
    for i in range(len(array)):
        for j in np.arange(0, i, 1):
            if array[j] > array[j+1]:
                temp = array[j]
                array[j] = array[j+1]
                array[j+1] = temp
    return array
def crop_center(img, cropx, cropy):
    
    #Function from 
    #https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    #print(startx)
    starty = y // 2 - (cropy // 2) 
    #print(starty)
    return img[starty:starty + cropy, startx:startx + cropx, ...]
def dualAGN_3DPSFFit(QSO, quasar_name, radius):
    startTime = time.time()
    #print(np.shape(QSO))
    #badval = []
    #badval.append(np.where(np.isnan(QSO)))
    #QSO = np.delete(QSO, badval)
    #fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,10))
    #ax = ax.flatten()
    imsize = len(QSO)
    logging.info(f"Now processing AGN: {quasar_name}")
    """Background subtraction algorithm"""
    QSO = np.ascontiguousarray(QSO)
    QSO = QSO.byteswap(inplace = True).newbyteorder()
    bkgd = sep.Background(QSO)
    QSO = QSO - bkgd
    Sources_Source_Extract = sep.extract(QSO, 1.5, err = bkgd.globalrms)
    numSources = len(Sources_Source_Extract)
    logging.info("Number of sources detected using SOURCE_EXTRACT: " + str(numSources))
    
    Y, X = np.indices(QSO.shape)
    xdata = np.vstack((X.ravel(), Y.ravel()))
    p0 = [np.max(QSO), imsize/2, imsize/2, 4.52 / (2 * np.sqrt(2 * np.log(2))), 4.52 / (2 * np.sqrt(2 * np.log(2)))]
    #The following function from lmfit will also model the Gaussian data and return the Chi-Squared value of the fit
    model = lmfit.models.Gaussian2dModel(nan_policy = 'omit')
    #This to create two Gaussians so that I can fit it to the data
    #General rule: freeze some parameters, fitting, loosening parameters, and then fit again
    #Step 1: Fit 1 Gaussian
    #Fit with a single Gaussian, take that fit, let the positions vary, but start where I have the highest intensity. Freeze sigma_x and sigma_y
    #Establish a sigma_a and a sigma_b for components A and B of the two Gaussian
    #Caution: the Gaussian may fit background galaxy light. Initially make sure I throw away all image where galactic structure is visible
    #For single point-sources, the normilization A should be very close to zero.
    #Keep varying until the model is able to pick up 
    params = model.make_params(amplitude = p0[0], centerx = p0[1], sigmax = p0[3], centery =p0[2], sigmay = p0[4])
    error = np.sqrt(np.abs(QSO+1))
    result = model.fit(QSO, x=X, y=Y, params=params, weights=1/error)
    fit = model.func(X, Y, **result.best_values)
    
    
    #The following code will fit two Gaussians to the data, as described in Prof. Urry's algoritm in plot3DData()
    gauss_1 = lmfit.models.Gaussian2dModel(prefix='g1_')
    paramsOne = gauss_1.make_params(amplitude = p0[0], centerx = p0[1], sigmax = p0[3], centery =p0[2], sigmay = p0[4])
    #paramsTwo['g2_amplitude'].set(value=p0[0], min=2)
    gauss_2 = lmfit.models.Gaussian2dModel(prefix='g2_')
    paramsOne.update(gauss_2.make_params())
    paramsOne['g2_amplitude'].set(value= p0[0]) 
    #paramsOne['g1_centerx'].set(p0[1], vary = False)
    #paramsOne['g1_centery'].set(p0[2], vary = False)
    paramsOne['g2_centerx'].set(value = p0[1]+5, min = p0[1]-radius, max = p0[1] + radius)
    paramsOne['g2_centery'].set(value = p0[2], min = p0[2]-radius, max = p0[2]+radius)
    #for xCenter and yCenter parameters, find the xCenter and yCenter of the first peak, and then vary it slightly in one of the two directions to see
    #how the fit function is able to fit to the second peak. 
    #paramsOne['g2_centerx'].set(value=imsize/3, min=imsize/3.5, max=imsize/1.5)
    paramsOne['g2_sigmax'].set(value=p0[3], min = 1)
    paramsOne['g2_sigmay'].set(value=p0[4], min = 1)
    paramsOne['g2_centery'].set(value=p0[2]+5)
    #paramsTwo['g2_amplitude'].set(value=p0[0], min=2)

    twinPeakModel = gauss_1 + gauss_2
    initial_values = twinPeakModel.eval(params, x=X, y=Y)
    output_values = twinPeakModel.fit(QSO, paramsOne, x=X, y=Y)

    
    #Error calculations
    """The following script implements the Henze-Zirkler test for multivariate normality, which determines whether 
    a data set follows the null hypothesis – that the data values are normally distributed – by taking the weighted integal of
    the squared modulus of the difference between the characteristic function of squared residuals and its limit as the data
    drops off. This function will return the p-value for this test. A low p-value will mean """
    #print(multivariate_normality(QSO[int(xCenter)-radius:int(xCenter)+radius, int(yCenter)-radius:int(yCenter)+radius], alpha = 0.05)) #this 0.05 is a placeholder value that I will change based on a better understanding of
    #alpha values
    prelimArray = [] #placeholder array to hold values for later use
    for key, values in output_values.best_values.items():
        prelimArray.append(values)
    #print(prelimArray)
    g1_amplitude = output_values.params['g1_amplitude'].value
    #print("g1_amplitude: " + str(g1_amplitude))
    g1_xcenter = output_values.params['g1_centerx'].value
    #print(g1_xcenter)
    g1_ycenter = output_values.params['g1_centery'].value
    #print(g1_ycenter)
    g1_sigmaX = output_values.params['g1_sigmax'].value
    g1_sigmaY = output_values.params['g1_sigmay'].value
    print("(g1_sigmaX, g1_sigmaY)" + str((g1_sigmaX, g1_sigmaY)))
    g1_fwhmx = output_values.params['g1_fwhmx'].value
    g1_fwhmy = output_values.params['g1_fwhmy'].value
    g1_height = output_values.params['g1_height'].value
    
    g2_amplitude = output_values.params['g2_amplitude'].value
    g2_xcenter = output_values.params['g2_centerx'].value
    g2_ycenter = output_values.params['g2_centery'].value
    g2_sigmaX = output_values.params['g2_sigmax'].value
    g2_sigmaY = output_values.params['g2_sigmay'].value
    g2_fwhmx = output_values.params['g2_fwhmx'].value
    g2_fwhmy = output_values.params['g2_fwhmy'].value
    g2_height = output_values.params['g2_height'].value
    reducedChiSquare = output_values.redchi
    chiSquare = output_values.chisqr
    distance_firstPass = np.sqrt((g1_xcenter - g2_xcenter)**2 + (g1_ycenter - g2_ycenter)**2)
    sigma_r_1_firstPass = np.sqrt((g1_sigmaX)**2 + (g1_sigmaY)**2)
    sigma_r_2_firstPass = np.sqrt((g2_sigmaX)**2 + (g2_sigmaY)**2)

    doubleNucleus = False
    if distance_firstPass >= 4.0:
        doubleNucleus = True
        print ("Two Sources Detected using Gaussian Fit")
    print("Original reduced chi squared value: " + str(reducedChiSquare))
    newFitNecessary = False
    p_value = 0
    fitParams = [g1_amplitude, (g1_xcenter/g2_xcenter), (g1_ycenter/g2_ycenter), g1_xcenter, g1_ycenter, g1_sigmaX, g1_sigmaY, g2_amplitude, g2_xcenter, g2_ycenter, g2_sigmaX, g2_sigmaY,
                 reducedChiSquare, doubleNucleus, newFitNecessary, p_value, numSources]
    
    
    #based on the 25 quasar sample and the rotational simulations. The cutoff for chi-squared values should be temporarily set to 100.
    #any fit with a chi-squared above 100 and/or a sigmax/sigmay > 1.2 will be flagged by the program as potentially having a double nucleus, and will be saved for further inspection
    
    """Following section re-performs fit if reduced chi-square value is > 0.06. Fits with large reduced chi-square values
    do not accurately reflect the positions of the centers of the quasars and thus are not useful in determining the if a double nucleus quasar exists"""
    placeholder = True # We want the best fit possible for candidate analysis, so we switch this parameter back on.
    if placeholder:
        start_time = time.time()
        #Fit a single Gaussian to the data and see if it accurately finds the center of the central QSO
        results_xcenter = result.params['centerx'].value
        results_ycenter = result.params['centery'].value
        results_sigmaX = result.params['sigmax'].value
        results_sigmaY = result.params['sigmay'].value
        results_amplitude = result.params['amplitude'].value
        #plt.show()
        newParams= gauss_1.make_params()
        newParams['g1_amplitude'].set(g1_amplitude, min = results_amplitude*0.2, max = results_amplitude*5, vary = True)
        newParams['g1_centerx'].set(len(X)/2, min = len(X)/2 - 2* radius, max = len(X)/2 + 2* radius, vary = True)
        newParams['g1_centery'].set(len(Y)/2, min = len(Y)/2 - 2 * radius, max = len(Y)/2 + 2 * radius, vary = True)
        #newParams['g1_sigmax'].set(results_sigmaX, min = 2.0, max = 4.0, vary = True)
        newParams['g1_sigmax'].set(2.6023513240476217, min = 2.0, max = 5.0, vary = False)
        newParams['g1_sigmay'].set(2.6023513240476217, min = 2.0, max = 5.0, vary = False)#Note: these are placeholder
        #values obtained preliminarily from the dataset. For a more rigorous fit, I will probably need to drive a better
        #feeder value for these sigma values.
        #newParams['g1_sigmay'].set(results_sigmaY, min = 2.0, max = 4.0, vary = True)
        gauss1_fit = gauss_1.fit(QSO, newParams, x=X, y=Y)
        newParams.update(gauss_2.make_params())
        newParams['g2_amplitude'].set(g1_amplitude, min = g1_amplitude*0.2, max = g1_amplitude*5, vary=True)
        newParams['g2_sigmax'].set(gauss1_fit.params['g1_sigmax'].value, min = gauss1_fit.params['g1_sigmax'].value - 0.02, max = gauss1_fit.params['g1_sigmax'].value + 0.002, vary = False)
        newParams['g2_sigmay'].set(gauss1_fit.params['g1_sigmay'].value, min = gauss1_fit.params['g1_sigmay'].value - 0.02, max = gauss1_fit.params['g1_sigmay'].value + 0.002, vary = False)
        newFits = []
        redChiFits = []
        #crazy idea to make sure that we get the right orientation of secondary fit each time. May take too much computing power:
        startTime = time.time()
        start = 0
        stop = 360
        step = 5
        i_values = np.arange(start, stop, step)
        # Calculate the corresponding values for 'newParams['g2_centerx']' and 'newParams['g2_centery']'
        g2_centerx_values = gauss1_fit.params['g1_centerx'].value + 5 * np.cos(i_values * (np.pi / 180.0))
        g2_centery_values = gauss1_fit.params['g1_centery'].value + 5 * np.sin(i_values * (np.pi / 180.0))
        # Calculate the 'min' and 'max' values for 'newParams['g2_centerx']' and 'newParams['g2_centery']'
        min_x_values = np.absolute(len(X) / 2) + 10
        max_x_values = len(X) / 2 + 2*radius * np.cos(i_values * np.pi / 180) + 1
        
        min_y_values = np.absolute(len(Y) / 2) + 10
        max_y_values = len(Y) / 2 + 2*radius * np.sin(i_values * np.pi / 180) + 1

        # Set the values in 'newParams' using broadcasting
        newFits = []
        redChiFits = []
        for temp in range(len(i_values)):
            startingtime = time.time()
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format("="*temp, len(i_values)-1, (100/(len(i_values)-1)*temp)))
            sys.stdout.flush()
            
            newParams['g2_centerx'].set(g2_centerx_values[temp], min=min_x_values, max=max_x_values[temp], vary=True)
            newParams['g2_centery'].set(g2_centery_values[temp], min=min_y_values, max=max_y_values[temp], vary=True)
            newFit = twinPeakModel.fit(QSO, newParams, x=X, y=Y, nan_policy='omit')
            while (newFit.params['g1_centerx'].value == newFit.params['g2_centerx']) and (newFit.params['g1_centery'].value == newFit.params['g2_centery']) and (np.abs(newFit.params['g1_centerx'].value - newFit.params['g2_centerx']) <= 5) and (np.abs(newFit.params['g1_centery'].value - newFit.params['g2_centery']) <= 5):
                newParams['g2_centerx'].set(g2_centerx_values[temp] + np.random.randint(0, radius)*np.cos(temp*np.pi/180.0), min = min_x_values, max = max_x_values, vary = True)
                newParams['g2_centery'].set(g2_centery_values[temp] + np.random.randint(0, radius)*np.sin(temp*np.pi/180.0), min = min_y_values, max = max_y_values, vary = True)
                newFit = twinPeakModel.fit(QSO, newParams, x=X, y=Y, nan_policy = 'omit')
            newFits.append(newFit)
            redChiFits.append(newFit.redchi)

        redChiFits = bubbleSort(redChiFits)
        lowestRedChi = redChiFits[0]
    
        loc = np.where(redChiFits == lowestRedChi)[0][0]
        print("New lowest chi squared = " + str(lowestRedChi))
        print("Secondary Fit took: " + str(time.time() - startTime) + " seconds")
        newFit = newFits[loc]
        g1_xCenter_refit = newFit.params['g1_centerx'].value
        
        g1_yCenter_refit = newFit.params['g1_centery'].value
        g2_xCenter_refit = newFit.params['g2_centerx'].value
        g2_yCenter_refit = newFit.params['g2_centery'].value
        g1_sigmaX_refit = newFit.params['g1_sigmax'].value
        g1_sigmaY_refit = newFit.params['g1_sigmay'].value
        g2_sigmaX_refit = newFit.params['g2_sigmax'].value
        g2_sigmaY_refit = newFit.params['g2_sigmay'].value
        distance_refit = np.sqrt((g1_xCenter_refit - g2_xCenter_refit)**2 + (g1_yCenter_refit - g2_yCenter_refit)**2)
        sigma_r_1_refit = np.sqrt((g1_sigmaX_refit)**2 + (g1_sigmaY_refit)**2)
        sigma_r_2_refit = np.sqrt((g2_sigmaX_refit)**2 + (g2_sigmaY_refit)**2)
        newFitNecessary = True

        plotFit = np.reshape(newFit.best_fit, np.shape(X))
        chiSquared2 = newFit.redchi
        #perform F-Test
        p_value = scipy.stats.f.cdf(output_values.best_fit, newFit.best_fit)
        print("p_value = " + str(p_value))
        alpha = 0.05 #standard value for alpha for an f-test
        if p_value > alpha:
            print("Secondary fit disproves null hypothesis, new fit is significant")
        fitParams[15] = p_value
        fitParams[12] = chiSquared2
        if (np.round(distance_refit, 3) >= 4.0 and np.round((sigma_r_1_refit/sigma_r_2_refit), 3) >=0.7 and np.round((sigma_r_1_refit/sigma_r_2_refit), 3) <=1.4):
            doubleNucleus = True
            fitParams[13] = True
            print("Two Sources Detected with refit")
        print("--- %s seconds ---" % (time.time() - start_time))
        
    print("--- %s seconds ---" % (time.time() - startTime))
    return fitParams

class testResults:
    def __init__(self, dataset, image_names, data_dir):
        #self.trained_model = trained_model
        self.dataset = dataset
        self.image_names = image_names
        #self.test_results = test_results
        self.quasars = []
        self.doubles = []
        self.double_names = []
        self.current_wd = data_dir
    def test(self, trained_model, filepath = None, fits_filepath = None, tang_png_filepath = None):
        filepath = f"{self.current_wd}binary_candidates/"
        fits_filepath = f"{self.current_wd}binary_candidates/binary_fits_files"
        tang_png_filepath = f"{self.current_wd}binary_candidates/tang_png_files/"
        """if isinstance(filepath, type(None)):
            filepath = f"{self.current_wd}binary_candidates/"
        elif isinstance(fits_filepath, type(None)):
            fits_filepath = f"{self.current_wd}binary_candidates/binary_fits_files"
        elif isinstance(tang_png_filepath, type(None)):
            tang_png_filepath = f"{self.current_wd}binary_candidates/tang_png_files/"
            """
        predictions = trained_model.predict(self.dataset, verbose = 1)
        label_names = ["empty_sky", "single_AGN", "dual_AGN", "merger"]
        self.test_results = []
        num_doubles = 0
        num_singles = 0
        header = ["object_ID", "label", "confidence"]
        with open(f"{self.current_wd}inf.csv", 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, prediction in tqdm(enumerate(predictions)):
                max_confidence = np.max(prediction)
                predicted_class = np.argmax(prediction)
                label = label_names[predicted_class]
                result = {'label': label, 'confidence': max_confidence}
                self.test_results.append(result)
                writer.writerow([self.image_names[i], label, max_confidence])
                #if label == "double AGN" and (self.image_names[i] != "QSO_220521_34_000009_6.fits" or self.image_names[i] != "QSO_220556_48_050507_1.fits" orself.image_names[i] != "QSO_220710_58_003840_1.fits" or self.image_names[i] != "QSO_221137_15_053921_9.fits" or self.image_names[i] != "QSO_221430_65_035432_0.fits"):
                if label == 'dual_AGN' or label == 'merger':
                    plt.figure()
                    self.doubles.append(self.dataset[i])
                    #plt.imshow(self.dataset[i], cmap = "gray_r", norm = mpl.colors.LogNorm())
                    plt.imshow(self.dataset[i], cmap = "gray_r", vmin = np.percentile(self.dataset[i], 1), vmax = np.percentile(self.dataset[i], 99))
                    plt.title(f"Predicted Label: {label}, Confidence: {max_confidence:.4f}")
                    plt.xlabel(self.image_names[i][:-4])
                    self.double_names.append(self.image_names[i])
    
                    if not exists(filepath + self.image_names[i]):
                        os.makedirs(filepath + self.image_names[i])
                    if not exists(fits_filepath):
                        os.makedirs(fits_filepath)
                    if not exists(tang_png_filepath):
                        os.makedirs(tang_png_filepath)
                    shutil.copy(self.image_names[i], filepath + self.image_names[i])
                    shutil.copy(self.image_names[i], fits_filepath)
                    plt.savefig(filepath + self.image_names[i][:-4] + "_.png")
                    #plt.show()
                    shutil.copy(filepath+self.image_names[i][:-4]+"_.png", tang_png_filepath)
                    #plt.axis('off')
                    plt.close()
                    num_doubles+=1
                else:
                    num_singles+=1
                    #plt.close()
            plt.close()
        df = pd.read_csv(f"{self.current_wd}inf.csv")
        CANDIDATES = set(["220906_91", "221115_06", "222057_44", "220718_43", "220811_56"])
        for _, row in df.iterrows():
            filename = row["object_ID"]
            for candidate in CANDIDATES:
                if candidate in filename:
                    print(row)
                    
        print("Number of double AGN: ", num_doubles)
        print("Number of single AGN: ", num_singles)
        return self.test_results

    def return_over_80_conf(self): #returns predictions that the model made with >80% confidence 
        over_80_counter = 0
        for i, item in enumerate(self.test_results):
            if item['confidence'] >= 0.8:
                dictionary = {'Quasar name': self.image_names[i], 'Predicted_label': item['label'], 'Confidence': item['confidence']}
                over_80_counter+=1
                self.quasars.append(dictionary)
        #print(self.quasars)
        print(over_80_counter)
        return self.quasars

    def plot_histogram(self):
        confidences = []
        selected_indices = []
        np_quasars = np.asarray(self.quasars)
        for item in self.test_results:
            confidences.append(item['confidence'])
        tang_quasar_names = [f"{self.current_wd}UNK_220718_43_001723_1.fits", f"{self.current_wd}QSO_220811_56_023830_1.fits", f"{self.current_wd}UNK_220906_91_004543_9.fits", f"{self.current_wd}UNK_221115_06_000030_9.fits", f"{self.current_wd}QSO_221227_74_005140_7.fits"]
        mask = np.array([quasar_dict.get("Quasar name") in tang_quasar_names for quasar_dict in np_quasars])
        selected_indices = np.where(mask)[0]
        
        selected_tang_binaries = np_quasars[selected_indices]
        tang_confidences = []
        for ind, k in enumerate(selected_tang_binaries):
            tang_confidences.append(k['Confidence'])
            if not exists(f"{self.current_wd}binary_candidates/tang_png_images/"):
                os.makedirs(f"{self.current_wd}binary_candidates/tang_png_images/")
            if not exists(f"{self.current_wd}binary_candidates/tang_png_images/png_plots"):
                os.makedirs(f"{self.current_wd}binary_candidates/tang_png_images/png_plots")
            shutil.copy(tang_quasar_names[ind], "f{self.current_wd}binary_candidates/tang_png_images/")
            shutil.copy(f"{self.current_wd}binary_candidates/" + tang_quasar_names[ind] + "_.png", f"{self.current_wd}binary_candidates/tang_png_images/")
        confidences = np.asarray(confidences)
        tang_confidences = np.asarray(tang_confidences)
        n, bins, patches = plt.hist(confidences, bins = np.arange(0.0, 1.1, 0.05), histtype = 'step', stacked = True, label = "All 2426 images")
        for count, x in zip(n, bins):
            plt.annotate(str(int(count)), xy=(x, count), xytext=(0, 5), textcoords='offset points', ha='center')
        n_1, bins_1, patches_1 = plt.hist(tang_confidences, bins = np.arange(0.0, 1.1, 0.05), histtype = 'step', bottom = confidences.min(), stacked = True, label = 'Tang Quasars')
        
        plt.title("Confidences Values of Quasars Labeled by BinaryQuasarDetector")
        plt.xlabel("Confidences (percentage)")
        plt.ylabel("Number of Quasars")
        plt.yscale('log')
        plt.legend(loc = 'upper left')
        plt.savefig("f{self.current_wd}/binary_candidates/conf_histogram.png")
        plt.show()


    def plot_mollweide(self, ra_array, dec_array, name_array):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='mollweide')
        for i in range(len(ra_rad)):
            plt.scatter(ra_array[i], dec_array[i], label=name_array[i], s=100, marker='o')
        plt.title("Mollweide Projection of Five Dual Candidates for Observation")
        plt.xlabel("Right Ascension [Deg]")
        plt.ylabel("Declination [Deg]")
        plt.legend(loc = 0, fontsize = 'x-small')
        plt.show()


    def find_distances(self):
        np_doubles = np.asarray(self.doubles)
        np_quasars = np.asarray(self.quasars)
        tang_quasar_names = [f"{self.current_wd}UNK_220718_43_001723_1.fits", f"{self.current_wd}QSO_220811_56_023830_1.fits", f"{self.current_wd}UNK_220906_91_004543_9.fits", f"{self.current_wd}UNK_221115_06_000030_9.fits", f"{self.current_wd}QSO_221227_74_005140_7.fits"]
        #mask = np.array([quasar_dict.get("Quasar name") in tang_quasar_names for quasar_dict in np_quasars])
        tang_indices = np.where(np.isin(self.double_names, tang_quasar_names))[0]
        #selected_indices = np.where(mask)[0]
        contour_destination_filepath = f"{self.current_wd}binary_candidates/contour_plots/"
        psf_destination_filepath = f"{self.current_wd}binary_candidates/psf_plots/"
        if not exists(contour_destination_filepath):
            os.makedirs(contour_destination_filepath)
        if not exists(psf_destination_filepath):
            os.makedirs(psf_destination_filepath)
        radius = 29 #emperically-tested initial radius for the Gaussian fitting routine.
        distances = []
        pixel_scale = 0.167 #HSC asec/pixel conversion.
        tang_distances = []
        secondary_quasar_pixel_coordinates = []
        quasar_fit_radii = []
        for index, QSO in tqdm(enumerate(np_doubles)):
            fitParams = dualAGN_3DPSFFit(QSO, self.double_names[index][29:-5], 30, 30, contour_destination_filepath, psf_destination_filepath, radius)
            #IRAFPhot_table = IRAFQuasarFinder(QSO, self.double_names[index][29:-5], 30, 30, contour_destination_filepath, psf_destination_filepath, radius)
            #print(IRAFPhot_table)
            distance_first_pass = fitParams[17]
            distances.append(distance_first_pass)
            g1_xcenter = fitParams[3]
            g1_ycenter = fitParams[4]
            g2_xcenter = fitParams[8]
            g2_ycenter = fitParams[9]
            g1_sigmax = fitParams[5]
            g1_sigmay = fitParams[6]
            g2_sigmax = fitParams[10]
            g2_sigmay = fitParams[11]

            g1_sigma_r = np.sqrt((g1_sigmax)**2 + g1_sigmay**2)
            g2_sigma_r = np.sqrt(g2_sigmax**2 + g2_sigmay**2)

            secondary_quasar_pixel_coordinates.append([(g1_xcenter, g1_ycenter), (g2_xcenter, g2_ycenter)])
            quasar_fit_radii.append((g1_sigma_r, g2_sigma_r))
            if fitParams[14] == True:
                distance_second_pass = fitParams[18]
                distances.append(distance_second_pass)
            if np.equal(index, tang_indices).any():
                if fitParams[14] == True:
                    tang_distance_second_pass = fitParams[18]
                    tang_distances.append(tang_distance_second_pass)
                else:
                    tang_distances.append(distance_first_pass)
            #Distances are in pixels. We need to convert to arcseconds via the HSC asec/pixel conversion.
        distances = np.asarray(distances)
        pixel_distances = distances
        distances = distances*pixel_scale
        tang_distances = np.asarray(tang_distances)
        tang_distances = tang_distances*pixel_scale
        plt.hist(distances, bins = np.arange(0.1, 3.0, 0.1), label = "Binary Candidates", histtype = "step", stacked = True, color = "midnightblue")
        plt.hist(tang_distances, bins = np.arange(0.1, 3.0, 0.1), label = "Tang et al. Candidates", histtype = "step", stacked = True, color = "lightsalmon")
        plt.xlabel("Separations [arcseconds]")
        plt.ylabel("Number of Binary Candidates")
        plt.title("Angular Separation of Binary Candidates")
        plt.grid()
        plt.legend(loc = 'best')
        plt.savefig("f{self.current_wd}/binary_candidates/distances.png")
        plt.show()
        return distances, tang_distances, pixel_distances, secondary_quasar_pixel_coordinates, np_doubles, quasar_fit_radii, tang_indices
    #def pos_to_wcs(self, pixel_x, pixel_y, wcs): #useful for when I need to find the coordinates of the secondary object.

    def find_physical_separations(self, ra, dec, z, angular_separation):
        #ra = ra % 180
        #ra[ra > 90] -= 180
        #cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
        cosmo = FlatLambdaCDM(H0 = 69.6 * u.km / u.s / u.Mpc, Om0 =  0.286, Tcmb0 = 2.725)
        comoving_distance = cosmo.comoving_distance(z)
        print(comoving_distance)
        print(type(comoving_distance))
        central_quasar = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=comoving_distance)
        # Calculate the physical separation by assuming the same coordinates for the second object
        #secondary_quasar = central_quasar.directional_offset_by(angular_separation*u.deg, 0*u.deg, physical_distance)# assuming that the secondary object is at the same redshift (and therefore is physically associated)
        #secondary_quasar = central_quasar.directional_offset_by(angular_separation*u.deg, physical_distance)# assuming that the secondary object is at the same redshift (and therefore is physically associated)
        #secondary_quasar = central_quasar.directional_offset_by(0*u.deg, angular_separation*u.deg)
        # Calculate the transverse comoving separation
        angular_separation_rad = angular_separation*(1.0/206264.80624709636)
        physical_separation = angular_separation_rad * comoving_distance.to(u.kpc)
        #transverse_comoving_separation = angular_separation *u.arcsec * comoving_distance.to(u.rad)
        #physical_separation = transverse_comoving_separation.to(u.Mpc)
        #transverse_comoving_separation = angular_separation * physical_distance.to(u.rad)
        #transverse_comoving_separation = angular_separation/60.0 *cosmo.kpc_comoving_per_arcmin(z)
        #transverse_comoving_separation.to(u.kpc/u.arcsec)
        #print(transverse_comoving_separation)
        print(physical_separation)
        # Calculate the physical separation
        #physical_separation = (transverse_comoving_separation) * u.arcmin.to(u.deg) * cosmo.kpc_proper_per_arcmin(z).to(u.Mpc)
        #physical_separation = (transf)
        #physical_separation = transverse_comoving_separation.to(u.Mpc)

        #physical_separation = central_quasar.separation_3d(secondary_quasar)
        # Print the result
        print(f"Angular Separation: {angular_separation} arcseconds")
        print(f"Redshift: {z}")
        #print(f"Physical Separation: {physical_separation:.2f} Mpc")
        return physical_separation.value

    def plot_physical_separation(self, physical_separations, tang_physical_separations = None):
        #z_array = np.asarray(zList)
        #ra_array = np.asarray(raList)
        #dec_array = np.asarray(dec_array)
        #plt.hist(physical_separations, bins = 'auto', histtype = "step", stacked = True, label = 'All 45 Binary Candidates')
        n, bins, patches = plt.hist(physical_separations, bins='auto', histtype="stepfilled", stacked=True, label='All 94 Binary Candidates', alpha = 0.8, color = "midnightblue")
        n1, bins1, patches1 = plt.hist(tang_physical_separations, bins = 'auto', label = "Tang et al. Candidates", histtype = "stepfilled", stacked = True, color = "lightsalmon")
        for count, x in zip(n, bins):
            plt.annotate(str(int(count)), xy=(x + 10, count), xytext=(0, 5), textcoords='offset points', ha='center')

        plt.xlabel("Physical Separation [kpc]")
        plt.ylabel("Number of Quasar Pairs")
        plt.title("Physical Separations of Dual AGN Candidates")
        plt.grid()
        plt.legend(loc = 'best')
        plt.savefig(f"{self.current_wd}binary_candidates/physical_distances.png")
        plt.show()

    def quasar_photometry(self, image, center_coords, secondary_coords, quasar_fit_radii, flux_mag_0):
        primary_radius, secondary_radius = quasar_fit_radii
        sigclip = SigmaClip(sigma = 3.0, maxiters = 10)
        center_x, center_y = center_coords
        secondary_x, secondary_y = secondary_coords
        aperture_1 = CircularAperture(center_coords, r = 2.5*primary_radius)
        annulus_1 = CircularAnnulus(center_coords, r_in = 10, r_out = 20) # adjust the size of the annulus according to how large
        #the quasars should be.
        #aper_stats_1 = ApertureStats(image, aperture_1, sigma_clip = None)
        #bkgd_stats = ApertureStats(image, annulus_1, sigma_clip = sigclip)
        #total_bkg = bkg_stats.median * aper_stats_1.sum_aper_area.value
        i#mage = image - bkgd_mean
        aperture_2 = CircularAperture(secondary_coords, r = 2.5*secondary_radius)
        #annulus_2 = CircularAnnulus(secondary_coords, r_in = 10, r_out = 20)
        #apertures = [aperture_1, aperture_2]
        #aper_stats = ApertureStats(image, aperture_2, sigma_clip = None)
        phot_table_1 = aperture_photometry(image, aperture_1)
        for col in phot_table_1.colnames:
            phot_table_1[col].info.format = '%.8g'  # for consistent table output
        adu_1 = phot_table_1['aperture_sum'].value[0]
        #print(adu_1)
        inst_mag_1 = ADUsToInstMag(adu_1, flux_mag_0)
        print("Instrumental magnitude 1: ", inst_mag_1)
        phot_table_2 = aperture_photometry(image, aperture_2)
        for col2 in phot_table_2.colnames:
            phot_table_2[col2].info.format = '%.8g'
        adu_2 = phot_table_2['aperture_sum'].value[0]
        #print(adu_2)
        inst_mag_2 = ADUsToInstMag(adu_2, flux_mag_0)
        print("Instrumental Magnitude of First Quasar: ", inst_mag_1)
        print("Instrumental Magnitude of Second Quasar: ", inst_mag_2)

        flux_ratio = 10**(-(inst_mag_1 - inst_mag_2)/2.5)
        if flux_ratio < 1: # to consistently compare the brighter quasar to the dimmer quasar in the pair
            flux_ratio = 1/flux_ratio
        mag_difference = np.abs(inst_mag_1 - inst_mag_2)
       
        plt.imshow(image, vmin = np.percentile(image, 1), vmax = np.percentile(image, 99))
        ap_patches_1 = aperture_1.plot(color='white', lw=2,
                          label='Photometry aperture 1')
        ap_patches_2 = aperture_2.plot(color = 'white', lw = 2, label = 'Photometry aperture 2')
        plt.legend()
        plt.show()
        print(phot_table_1)
        print(phot_table_2)
        return inst_mag_1, inst_mag_2, flux_ratio, mag_difference
        #phot_table_1 = aperture_photometry(image, annulus_1)


class loadModelClass:
    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
    def f1_score(self, y_true, y_pred):
        precision = tf.keras.metrics.Precision(thresholds=0.5)
        recall = tf.keras.metrics.Recall(thresholds=0.5)
        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)
        precision_result = precision.result()
        recall_result = recall.result()
        f1 = 2 * ((precision_result * recall_result) / (precision_result + recall_result + tf.keras.backend.epsilon()))
        precision.reset_states()
        recall.reset_states()
        return f1
    def MCC(self, y_true, y_pred):
        true_positives = tf.keras.metrics.TruePositives()
        true_negatives = tf.keras.metrics.TrueNegatives()
        false_positives = tf.keras.metrics.FalsePositives()
        false_negatives = tf.keras.metrics.FalseNegatives()
        true_positives.update_state(y_true, y_pred)
        true_negatives.update_state(y_true, y_pred)
        false_positives.update_state(y_true, y_pred)
        false_negatives.update_state(y_true, y_pred)
        true_positives_result = true_positives.result()
        true_negatives_result = true_negatives.result()
        false_positives_result = false_positives.result()
        false_negatives_result = false_negatives.result()
        mcc = ((true_positives_result*true_negatives_result) -
               (false_positives_result-false_negatives_result))/np.sqrt((true_positives_result + false_positives_result)*(true_positives_result + false_negatives_result)*(true_negatives_result + false_positives_result)*(true_negatives_result + false_negatives_result))
        return mcc

    def load_model(self, initial_learning_rate):
        trained_model = models.load_model(self.path_to_model, compile=False)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                      decay_steps=10000,decay_rate=0.9)
        opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
        trained_model.compile(optimizer = opt, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               self.f1_score, self.MCC], run_eagerly=False)
        return trained_model
